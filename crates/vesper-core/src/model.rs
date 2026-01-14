//! VesperLM Main Model
//! 
//! Transformer architecture with FlyLoRA and ERA activation

use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, Embedding, LayerNorm, Linear, VarBuilder};

use crate::attention::MultiHeadAttention;
use crate::config::VesperConfig;
use crate::era::ERAActivation;
use crate::flylora::FlyLoRALinear;

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

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        // Final layer norm
        hidden_states = self.final_norm.forward(&hidden_states)?;

        // Language modeling head
        self.lm_head.forward(&hidden_states)
    }

    pub fn config(&self) -> &VesperConfig {
        &self.config
    }
}

struct TransformerLayer {
    attention: MultiHeadAttention,
    ffn: FeedForward,
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

        let ffn = FeedForward::new(config, vb.pp("ffn"))?;

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

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm attention with residual
        let normed = self.attention_norm.forward(hidden_states)?;
        let attn_out = self.attention.forward(&normed, attention_mask)?;
        let hidden_states = hidden_states.add(&attn_out)?;

        // Pre-norm FFN with residual
        let normed = self.ffn_norm.forward(&hidden_states)?;
        let ffn_out = self.ffn.forward(&normed)?;
        hidden_states.add(&ffn_out)
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
            temperature: config.era_temperature,
            entropy_weight: config.era_entropy_weight,
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
    use candle_core::Device;

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
        let logits = model.forward(&input_ids, None)?;

        assert_eq!(logits.dims(), &[2, 32, config.vocab_size]);
        Ok(())
    }
}
