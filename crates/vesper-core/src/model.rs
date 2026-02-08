//! VesperLM Main Model
//!
//! Transformer architecture with FlyLoRA and ERA activation

use candle_core::{D, DType, Module, Result, Tensor, Var};
use candle_nn::{embedding, Embedding, LayerNorm, Linear, VarBuilder};

use crate::attention::MultiHeadAttention;
use crate::config::VesperConfig;
use crate::era::ERAActivation;
use crate::flylora::FlyLoRALinear;
use crate::moe::MoELayer;

/// Data produced by a checkpointed forward pass for multi-phase backward.
pub struct CheckpointData {
    /// Boundary activations as Vars at each segment start.
    pub boundary_vars: Vec<Var>,
    /// Precomputed 4D attention mask (reused for all segment recomputations).
    pub mask_4d: Option<Tensor>,
    /// Layer ranges per segment: (start_inclusive, end_exclusive).
    pub segment_ranges: Vec<(usize, usize)>,
    /// Raw output of the last segment (before final_norm + lm_head), detached.
    pub last_hidden: Tensor,
    /// MoE aux_loss sum from all segments (detached, for logging only).
    pub total_aux_loss: Option<Tensor>,
}

/// Boundary hidden states for layer-by-layer backward.
///
/// Each boundary is a detached tensor — the computation graph for each layer is
/// freed immediately after forward. During backward, each layer is recomputed
/// one at a time, keeping only ~1 layer's intermediates in memory.
///
/// Memory: model + optimizer + 1 layer's graph ≈ 12-15GB for batch 10.
/// vs full autograd: model + optimizer + ALL layers' graphs ≈ 70GB.
pub struct LayerBoundaries {
    /// Detached hidden states: boundaries[i] = input to layer i,
    /// boundaries[num_layers] = input to head (output of last layer).
    pub boundaries: Vec<Tensor>,
    /// Precomputed 4D attention mask (reused during layer backward).
    pub mask_4d: Option<Tensor>,
    /// Number of transformer layers.
    pub num_layers: usize,
}

pub struct VesperLM {
    config: VesperConfig,
    embeddings: Embedding,
    layers: Vec<TransformerLayer>,
    final_norm: LayerNorm,
    lm_head: Linear,
}

impl VesperLM {
    pub fn new(config: VesperConfig, vb: VarBuilder) -> Result<Self> {
        config.validate().map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embeddings"),
        )?;

        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            layers.push(TransformerLayer::new(
                &config,
                vb.pp(&format!("layers.{}", layer_idx)),
            )?);
        }

        let final_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_norm"),
        )?;

        let lm_head = candle_nn::linear(
            config.hidden_size,
            config.vocab_size,
            vb.pp("lm_head"),
        )?;

        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let hidden_states = self.embeddings.forward(input_ids)?;
        let mask_4d = Self::compute_mask_4d(attention_mask, hidden_states.dtype())?;

        let (hidden_states, total_aux_loss) = self.forward_segment(
            &hidden_states, mask_4d.as_ref(), 0, self.layers.len(),
        )?;

        // Manual layer norm + lm_head (forward_head uses standard ops so gradients
        // flow through to all layers — the fused kernel severs the graph)
        let logits = self.forward_head(&hidden_states)?;
        Ok((logits, total_aux_loss))
    }

    /// Checkpointed forward pass for memory-efficient training.
    ///
    /// Splits layers into `config.gradient_checkpoint_segments` segments.
    /// At each boundary, copies hidden_states into a Var (breaking the graph).
    /// Only the last segment retains a live computation graph.
    /// Earlier segments' activations are freed after copying the boundary.
    ///
    /// The caller must run multi-phase backward using recompute_segment().
    pub fn forward_checkpointed(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<CheckpointData> {
        let num_segments = self.config.gradient_checkpoint_segments.max(1);
        let num_layers = self.layers.len();
        let layers_per_seg = (num_layers + num_segments - 1) / num_segments;

        // Build segment ranges
        let mut segment_ranges = Vec::new();
        let mut start = 0;
        while start < num_layers {
            let end = (start + layers_per_seg).min(num_layers);
            segment_ranges.push((start, end));
            start = end;
        }
        let actual_segments = segment_ranges.len();

        // Embeddings
        let embedded = self.embeddings.forward(input_ids)?;
        let dtype = embedded.dtype();
        let mask_4d = Self::compute_mask_4d(attention_mask, dtype)?;

        // boundary_vars[0] = Var wrapping embedding output
        let mut boundary_vars: Vec<Var> = vec![Var::from_tensor(&embedded)?];

        let mut total_aux: Option<Tensor> = None;

        for (seg_idx, &(layer_start, layer_end)) in segment_ranges.iter().enumerate() {
            let is_last = seg_idx == actual_segments - 1;
            let seg_input = boundary_vars[seg_idx].as_tensor();

            let (seg_output, seg_aux) = self.forward_segment(
                seg_input, mask_4d.as_ref(), layer_start, layer_end,
            )?;

            // Accumulate aux_loss (detached, for logging only)
            if let Some(a) = seg_aux {
                let a_det = a.detach();
                total_aux = Some(match total_aux {
                    Some(acc) => acc.add(&a_det)?,
                    None => a_det,
                });
            }

            if is_last {
                // Detach last segment output — backward phase will recompute via forward_head.
                // No live computation graph stored → frees all segment activations.
                return Ok(CheckpointData {
                    boundary_vars,
                    mask_4d,
                    segment_ranges,
                    last_hidden: seg_output.detach(),
                    total_aux_loss: total_aux,
                });
            } else {
                // Copy data into a new Var, freeing this segment's activations
                boundary_vars.push(Var::from_tensor(&seg_output)?);
            }
        }
        unreachable!("segment loop must hit is_last")
    }

    /// Recompute a segment's forward pass from its boundary Var.
    /// Used during multi-phase backward.
    ///
    /// For segment 0: recomputes from input_ids through embeddings (captures embedding grads).
    /// For segment N>0: runs from boundary_vars[seg] (uses Var as leaf).
    pub fn recompute_segment(
        &self,
        seg_idx: usize,
        input_ids: &Tensor,
        boundary_vars: &[Var],
        mask_4d: Option<&Tensor>,
        segment_ranges: &[(usize, usize)],
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (layer_start, layer_end) = segment_ranges[seg_idx];

        if seg_idx == 0 {
            // Recompute from input_ids through embeddings to capture embedding gradients
            let embedded = self.embeddings.forward(input_ids)?;
            self.forward_segment(&embedded, mask_4d, layer_start, layer_end)
        } else {
            // Run from boundary Var (a variable leaf whose gradient backward tracks)
            let seg_input = boundary_vars[seg_idx].as_tensor();
            self.forward_segment(seg_input, mask_4d, layer_start, layer_end)
        }
    }

    /// Apply final layer norm + language modeling head to hidden states.
    /// Used by checkpointed backward to recompute logits from a segment's output.
    ///
    /// Uses manual layer norm (standard ops) instead of the fused kernel so that
    /// gradients can flow through to the input Var. The fused kernel
    /// (`apply_op3_no_bwd`) severs the computation graph.
    pub fn forward_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Manual layer norm using standard Candle ops (full backward support).
        // Reads weight/bias from self.final_norm to stay checkpoint-compatible.
        let x_dtype = hidden_states.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_f64 = self.config.hidden_size as f64;
        let x = hidden_states.to_dtype(internal_dtype)?;
        let mean = (x.sum_keepdim(D::Minus1)? / hidden_f64)?;
        let x = x.broadcast_sub(&mean)?;
        let var = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_f64)?;
        let x_normed = x.broadcast_div(&(var + self.config.layer_norm_eps)?.sqrt()?)?;
        let normed = x_normed.to_dtype(x_dtype)?.broadcast_mul(self.final_norm.weight())?;
        let normed = match self.final_norm.bias() {
            Some(bias) => normed.broadcast_add(bias)?,
            None => normed,
        };
        self.lm_head.forward(&normed)
    }

    pub fn config(&self) -> &VesperConfig {
        &self.config
    }

    /// Forward through all layers, detaching at each boundary.
    ///
    /// Unlike `forward()` which builds a computation graph spanning ALL layers
    /// (consuming ~26GB for 24-layer 1B model), this method detaches after each
    /// layer so intermediate tensors are freed immediately. Only the boundary
    /// hidden states (~0.5GB for batch 10) are retained.
    ///
    /// The caller performs backward layer-by-layer using `recompute_layer()`.
    pub fn forward_with_boundaries(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<LayerBoundaries> {
        let embedded = self.embeddings.forward(input_ids)?;
        let mask_4d = Self::compute_mask_4d(attention_mask, embedded.dtype())?;

        let num_layers = self.layers.len();
        let mut boundaries = Vec::with_capacity(num_layers + 1);

        // boundaries[0] = input to layer 0 (embedding output, detached)
        let mut h = embedded.detach();
        boundaries.push(h.clone());

        for layer in &self.layers {
            let (new_h, _aux) = layer.forward(&h, mask_4d.as_ref())?;
            // Detach: frees this layer's computation graph (attention scores, FFN intermediates)
            h = new_h.detach();
            boundaries.push(h.clone());
        }
        // boundaries[num_layers] = output of last layer = input to head

        Ok(LayerBoundaries {
            boundaries,
            mask_4d,
            num_layers,
        })
    }

    /// Recompute a single layer's forward pass from its boundary input.
    /// Used during layer-by-layer backward to rebuild the computation graph
    /// for just one layer at a time.
    pub fn recompute_layer(
        &self,
        layer_idx: usize,
        input: &Tensor,
        mask_4d: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        self.layers[layer_idx].forward(input, mask_4d)
    }

    /// Forward through embeddings only.
    /// Used during layer 0 backward to capture embedding parameter gradients.
    pub fn embeddings_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embeddings.forward(input_ids)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Compute 4D causal+padding attention mask from an optional 2D mask.
    fn compute_mask_4d(attention_mask: Option<&Tensor>, dtype: DType) -> Result<Option<Tensor>> {
        let mask = match attention_mask {
            Some(m) => m,
            None => return Ok(None),
        };
        let dims = mask.dims();
        if dims.len() != 2 {
            return Ok(Some(mask.clone()));
        }
        let (_batch_size, seq_len) = (dims[0], dims[1]);

        // Causal mask
        let mut causal_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                causal_data[i * seq_len + j] = -1e9;
            }
        }
        let causal = Tensor::from_vec(causal_data, (1, 1, seq_len, seq_len), mask.device())?;
        let causal = causal.to_dtype(dtype)?;

        // Padding mask: [batch, seq] -> [batch, 1, 1, seq]
        let padding_mask = mask.to_dtype(dtype)?;
        let ones = Tensor::ones_like(&padding_mask)?;
        let padding_mask = (ones.sub(&padding_mask)? * (-1e9 as f64))?;
        let padding_mask = padding_mask.unsqueeze(1)?.unsqueeze(1)?;

        // Combine: causal + padding
        let combined = causal.broadcast_add(&padding_mask)?;
        Ok(Some(combined))
    }

    /// Run a range of transformer layers, accumulating aux_loss.
    fn forward_segment(
        &self,
        hidden_states: &Tensor,
        mask_4d: Option<&Tensor>,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mut h = hidden_states.clone();
        let mut total_aux: Option<Tensor> = None;

        for idx in layer_start..layer_end {
            let (new_h, aux) = self.layers[idx].forward(&h, mask_4d)?;
            h = new_h;
            if let Some(a) = aux {
                total_aux = Some(match total_aux {
                    Some(acc) => acc.add(&a)?,
                    None => a,
                });
            }
        }
        Ok((h, total_aux))
    }
}

enum FFNLayer {
    Standard(FeedForward),
    MoE(MoELayer),
}

struct TransformerLayer {
    attention: MultiHeadAttention,
    ffn: FFNLayer,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

impl TransformerLayer {
    fn new(config: &VesperConfig, vb: VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(
            config.hidden_size,
            config.num_heads,
            config.max_position_embeddings,
            config.rope_theta,
            vb.pp("attention"),
        )?;

        let ffn = if config.moe_enabled {
            FFNLayer::MoE(MoELayer::new(config, vb.pp("ffn"))?)
        } else {
            FFNLayer::Standard(FeedForward::new(config, vb.pp("ffn"))?)
        };

        // Fused LayerNorm (memory efficient). Gradients bypass these via residual connections.
        let attention_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attention_norm"),
        )?;

        let ffn_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("ffn_norm"),
        )?;

        Ok(Self {
            attention,
            ffn,
            attention_norm,
            ffn_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        // Pre-norm attention with residual
        let normed = self.attention_norm.forward(hidden_states)?;
        let attn_out = self.attention.forward(&normed, attention_mask)?;
        let hidden_states = hidden_states.add(&attn_out)?;

        // Pre-norm FFN with residual
        let normed = self.ffn_norm.forward(&hidden_states)?;
        let (ffn_out, aux_loss) = match &self.ffn {
            FFNLayer::Standard(ffn) => (ffn.forward(&normed)?, None),
            FFNLayer::MoE(moe) => {
                let (out, aux) = moe.forward(&normed)?;
                (out, Some(aux))
            }
        };
        Ok((hidden_states.add(&ffn_out)?, aux_loss))
    }
}

struct FeedForward {
    gate_proj: FlyLoRALinear,
    up_proj: FlyLoRALinear,
    down_proj: Linear,
    era: ERAActivation,
}

impl FeedForward {
    fn new(config: &VesperConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = FlyLoRALinear::new(
            config.hidden_size,
            config.intermediate_size,
            config.flylora_rank,
            config.flylora_alpha,
            config.flylora_sparsity,
            vb.pp("gate_proj"),
        )?;

        let up_proj = FlyLoRALinear::new(
            config.hidden_size,
            config.intermediate_size,
            config.flylora_rank,
            config.flylora_alpha,
            config.flylora_sparsity,
            vb.pp("up_proj"),
        )?;

        let down_proj = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;

        let era = ERAActivation::new(crate::era::ERAConfig {
            gamma: config.era_gamma,
        });

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            era,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // GLU-style FFN with ERA activation
        let gate = self.gate_proj.forward(hidden_states)?;
        let gate_activated = self.era.forward(&gate)?;
        
        let up = self.up_proj.forward(hidden_states)?;
        let gated = gate_activated.mul(&up)?;
        
        self.down_proj.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_model_creation() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = VesperConfig::tiny();
        let model = VesperLM::new(config.clone(), vb)?;

        assert_eq!(model.config().hidden_size, config.hidden_size);
        Ok(())
    }

    #[test]
    fn test_model_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = VesperConfig::tiny();
        let model = VesperLM::new(config.clone(), vb)?;

        let input_ids = Tensor::zeros((2, 32), DType::U32, &device)?;
        let (logits, aux_loss) = model.forward(&input_ids, None)?;

        assert_eq!(logits.dims(), &[2, 32, config.vocab_size]);
        assert!(aux_loss.is_none()); // No MoE, no aux loss
        Ok(())
    }

    #[test]
    fn test_moe_model_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = VesperConfig::tiny().with_moe(4, 2);
        let model = VesperLM::new(config.clone(), vb)?;

        let input_ids = Tensor::zeros((2, 16), DType::U32, &device)?;
        let (logits, aux_loss) = model.forward(&input_ids, None)?;

        assert_eq!(logits.dims(), &[2, 16, config.vocab_size]);
        assert!(aux_loss.is_some()); // MoE enabled → aux loss present
        Ok(())
    }
}
