//! Mixture of Experts (MoE) Layer
//!
//! Replaces the standard FFN with N expert FFNs and a learned router.
//! Each token is routed to the top-K experts, with outputs weighted
//! by normalized router probabilities.
//!
//! Includes Switch Transformer load balancing auxiliary loss to prevent
//! expert collapse.

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use crate::config::VesperConfig;
use crate::era::{ERAActivation, ERAConfig};
use crate::flylora::FlyLoRALinear;

/// Single expert FFN — identical architecture to the standard FeedForward.
/// GLU-style: gate_proj(FlyLoRA) → ERA → mul(up_proj(FlyLoRA)) → down_proj(Linear)
pub struct ExpertFFN {
    gate_proj: FlyLoRALinear,
    up_proj: FlyLoRALinear,
    down_proj: Linear,
    era: ERAActivation,
}

impl ExpertFFN {
    pub fn new(config: &VesperConfig, vb: VarBuilder) -> Result<Self> {
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

        let era = ERAActivation::new(ERAConfig {
            gamma: config.era_gamma,
        });

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            era,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(hidden_states)?;
        let gate_activated = self.era.forward(&gate)?;
        let up = self.up_proj.forward(hidden_states)?;
        let gated = gate_activated.mul(&up)?;
        self.down_proj.forward(&gated)
    }
}

/// Mixture of Experts layer with top-K routing and load balancing loss.
pub struct MoELayer {
    router: Linear,
    experts: Vec<ExpertFFN>,
    num_experts: usize,
    top_k: usize,
}

impl MoELayer {
    pub fn new(config: &VesperConfig, vb: VarBuilder) -> Result<Self> {
        let router = candle_nn::linear(
            config.hidden_size,
            config.moe_num_experts,
            vb.pp("router"),
        )?;

        let mut experts = Vec::with_capacity(config.moe_num_experts);
        for i in 0..config.moe_num_experts {
            experts.push(ExpertFFN::new(config, vb.pp(&format!("experts.{}", i)))?);
        }

        Ok(Self {
            router,
            experts,
            num_experts: config.moe_num_experts,
            top_k: config.moe_top_k,
        })
    }

    /// Forward pass: route tokens to top-K experts, return (output, aux_loss).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let device = x.device();

        // 1. Router: softmax over expert logits
        let router_logits = self.router.forward(x)?; // [B, S, N]
        let router_probs = candle_nn::ops::softmax(&router_logits, D::Minus1)?; // [B, S, N]

        // 2. Top-K routing: sort descending, take first K
        let router_probs_c = router_probs.contiguous()?;
        let sorted_indices = router_probs_c.arg_sort_last_dim(false)?; // [B, S, N] U32 desc
        let top_k_indices = sorted_indices.narrow(2, 0, self.top_k)?.contiguous()?; // [B, S, K]
        let top_k_probs = router_probs.gather(&top_k_indices, 2)?; // [B, S, K]

        // Normalize top-K weights to sum to 1
        let normalizer = top_k_probs.sum_keepdim(2)?;
        let top_k_weights = top_k_probs.broadcast_div(&normalizer)?; // [B, S, K]

        // 3. Compute weighted expert outputs
        let mut output = Tensor::zeros_like(x)?;

        for expert_id in 0..self.num_experts {
            // Build combined weight for this expert across all K slots
            let mut expert_weight = Tensor::zeros((batch_size, seq_len), DType::F32, device)?;
            for k in 0..self.top_k {
                let idx_k = top_k_indices.narrow(2, k, 1)?.squeeze(2)?; // [B, S] U32
                let w_k = top_k_weights.narrow(2, k, 1)?.squeeze(2)?; // [B, S] F32

                // Mask: 1 where this token selected expert_id at slot k
                let mask = idx_k.eq(expert_id as u32)?.to_dtype(DType::F32)?; // [B, S]
                expert_weight = (expert_weight + mask.mul(&w_k)?)?;
            }

            // Skip expert if no tokens routed to it
            let weight_sum = expert_weight.sum_all()?.to_scalar::<f32>()?;
            if weight_sum < 1e-6 {
                continue;
            }

            // Run expert and weight its output
            let expert_out = self.experts[expert_id].forward(x)?; // [B, S, H]
            let w_expanded = expert_weight
                .unsqueeze(2)?
                .broadcast_as(expert_out.shape())?; // [B, S, H]
            output = (output + expert_out.mul(&w_expanded)?)?;
        }

        // 4. Load balancing auxiliary loss
        let aux_loss = self.load_balancing_loss(&router_probs, &top_k_indices)?;

        Ok((output, aux_loss))
    }

    /// Switch Transformer load balancing loss: N × Σᵢ(fᵢ × Pᵢ)
    /// where fᵢ = fraction of tokens routed to expert i,
    ///       Pᵢ = mean router probability for expert i.
    fn load_balancing_loss(
        &self,
        router_probs: &Tensor, // [B, S, N]
        top_k_indices: &Tensor, // [B, S, K]
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = router_probs.dims3()?;
        let total_assignments = (batch_size * seq_len * self.top_k) as f64;

        // P_i: mean router probability per expert — [N]
        // Mean over batch and seq dims
        let p_mean = router_probs.mean(0)?.mean(0)?; // [N]

        // f_i: fraction of token-slots assigned to each expert — [N]
        // Build assignment counts on GPU using eq + sum
        let mut expert_fractions = Vec::with_capacity(self.num_experts);
        for expert_id in 0..self.num_experts {
            let mask = top_k_indices.eq(expert_id as u32)?.to_dtype(DType::F32)?;
            let count = mask.sum_all()?; // scalar
            let fraction = (count / total_assignments)?;
            expert_fractions.push(fraction);
        }

        // Stack fractions into [N] tensor
        let f_tensor = Tensor::stack(&expert_fractions, 0)?; // [N]

        // aux_loss = N × Σ(f_i × P_i)
        let products = f_tensor.mul(&p_mean)?;
        let sum = products.sum_all()?;
        sum * (self.num_experts as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_expert_ffn_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = VesperConfig::tiny();
        let expert = ExpertFFN::new(&config, vb)?;

        let input = Tensor::randn(0f32, 1.0, (2, 16, config.hidden_size), &device)?;
        let output = expert.forward(&input)?;

        assert_eq!(output.dims(), input.dims());
        Ok(())
    }

    #[test]
    fn test_moe_layer_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = VesperConfig::tiny().with_moe(4, 2);
        let moe = MoELayer::new(&config, vb)?;

        let input = Tensor::randn(0f32, 1.0, (2, 16, config.hidden_size), &device)?;
        let (output, aux_loss) = moe.forward(&input)?;

        assert_eq!(output.dims(), input.dims());
        assert_eq!(aux_loss.dims().len(), 0); // scalar
        let aux_val = aux_loss.to_scalar::<f32>()?;
        assert!(aux_val >= 0.0, "aux_loss should be non-negative, got {}", aux_val);
        Ok(())
    }

    #[test]
    fn test_moe_config_validation() {
        let mut config = VesperConfig::tiny();
        config.moe_enabled = true;
        config.moe_top_k = 10;
        config.moe_num_experts = 4;
        assert!(config.validate().is_err());

        config.moe_top_k = 2;
        assert!(config.validate().is_ok());
    }
}
