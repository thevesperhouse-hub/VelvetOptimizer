//! VesperLM Main Model
//! 
//! Transformer architecture with FlyLoRA and ERA activation

use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, Embedding, LayerNorm, Linear, VarBuilder};

use crate::attention::MultiHeadAttention;
use crate::config::VesperConfig;
use crate::era::ERAActivation;
use crate::flylora::FlyLoRALinear;
use crate::moe::MoELayer;

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
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Convert 2D attention mask [batch, seq] to 4D causal mask [batch, 1, seq, seq]
        let mask_4d = if let Some(mask) = attention_mask {
            let dims = mask.dims();
            if dims.len() == 2 {
                let (_batch_size, seq_len) = (dims[0], dims[1]);
                // Create causal mask
                let mut causal_data = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        causal_data[i * seq_len + j] = -1e9;
                    }
                }
                let causal = Tensor::from_vec(causal_data, (1, 1, seq_len, seq_len), mask.device())?;

                // Convert padding mask: [batch, seq] -> [batch, 1, 1, seq]
                // 0 in mask -> -1e9, 1 in mask -> 0 (using -1e9 instead of -inf to avoid 0*-inf=NaN)
                let padding_mask = mask.to_dtype(candle_core::DType::F32)?;
                let ones = Tensor::ones_like(&padding_mask)?;
                let padding_mask = (ones.sub(&padding_mask)? * (-1e9 as f64))?;
                let padding_mask = padding_mask.unsqueeze(1)?.unsqueeze(1)?;

                // Combine: causal + padding
                let combined = causal.broadcast_add(&padding_mask)?;
                Some(combined)
            } else {
                Some(mask.clone())
            }
        } else {
            None
        };

        // Pass through transformer layers, accumulating MoE auxiliary losses
        let mut total_aux_loss: Option<Tensor> = None;

        for layer in &self.layers {
            let (new_hidden, aux_loss) = layer.forward(&hidden_states, mask_4d.as_ref())?;
            hidden_states = new_hidden;

            if let Some(aux) = aux_loss {
                total_aux_loss = Some(match total_aux_loss {
                    Some(acc) => acc.add(&aux)?,
                    None => aux,
                });
            }
        }

        // Final layer norm
        hidden_states = self.final_norm.forward(&hidden_states)?;

        // Language modeling head
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok((logits, total_aux_loss))
    }

    pub fn config(&self) -> &VesperConfig {
        &self.config
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
