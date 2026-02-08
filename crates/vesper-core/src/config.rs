//! VesperLM Configuration
//! 
//! Simplified config focusing on core features

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VesperConfig {
    // Model architecture
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,

    // FlyLoRA config
    pub flylora_rank: usize,
    pub flylora_alpha: f64,
    pub flylora_sparsity: f32,

    // ERA activation: γ for GELU(x) + γ * softplus(x)
    pub era_gamma: f64,

    // Training
    pub dropout: f64,
    pub layer_norm_eps: f64,

    // Attention
    pub use_flash_attn: bool,
    pub rope_theta: f64,

    // Mixture of Experts
    pub moe_enabled: bool,
    pub moe_num_experts: usize,
    pub moe_top_k: usize,
    pub moe_aux_loss_weight: f64,

    // Gradient checkpointing: split layers into N segments, recompute during backward.
    // 0 = disabled, recommended: 4 or 6 for 24-layer models.
    pub gradient_checkpoint_segments: usize,
}

impl Default for VesperConfig {
    fn default() -> Self {
        Self {
            // Base architecture (small model)
            vocab_size: 32000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 4096,

            // FlyLoRA defaults
            flylora_rank: 16,
            flylora_alpha: 32.0,
            flylora_sparsity: 0.25,

            // ERA defaults
            era_gamma: 0.1,

            // Training defaults
            dropout: 0.1,
            layer_norm_eps: 1e-6,

            // Attention defaults
            use_flash_attn: true,
            rope_theta: 10000.0,

            // MoE defaults (disabled)
            moe_enabled: false,
            moe_num_experts: 8,
            moe_top_k: 2,
            moe_aux_loss_weight: 0.01,

            // Gradient checkpointing (disabled)
            gradient_checkpoint_segments: 0,
        }
    }
}

impl VesperConfig {
    /// Create config for different model sizes
    pub fn tiny() -> Self {
        Self {
            hidden_size: 256,
            num_layers: 6,
            num_heads: 4,
            intermediate_size: 1024,
            ..Default::default()
        }
    }

    pub fn small() -> Self {
        Self {
            hidden_size: 512,
            num_layers: 8,
            num_heads: 8,
            intermediate_size: 2048,
            ..Default::default()
        }
    }

    pub fn medium() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            ..Default::default()
        }
    }

    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
            ..Default::default()
        }
    }

    /// ~1B parameter configuration for cloud training (A100)
    pub fn xlarge() -> Self {
        Self {
            hidden_size: 2048,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 5504,
            max_position_embeddings: 4096,
            ..Default::default()
        }
    }

    /// Tiny model with MoE (4 experts, top-2)
    pub fn tiny_moe() -> Self {
        Self {
            moe_enabled: true,
            moe_num_experts: 4,
            moe_top_k: 2,
            moe_aux_loss_weight: 0.01,
            ..Self::tiny()
        }
    }

    /// Medium model with MoE (8 experts, top-2)
    pub fn medium_moe() -> Self {
        Self {
            moe_enabled: true,
            moe_num_experts: 8,
            moe_top_k: 2,
            moe_aux_loss_weight: 0.01,
            ..Self::medium()
        }
    }

    /// Large model with MoE (16 experts, top-2)
    pub fn large_moe() -> Self {
        Self {
            moe_enabled: true,
            moe_num_experts: 16,
            moe_top_k: 2,
            moe_aux_loss_weight: 0.01,
            ..Self::large()
        }
    }

    /// Builder: set vocab_size (must match tokenizer)
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Builder: enable MoE with given expert count and top-k
    pub fn with_moe(mut self, num_experts: usize, top_k: usize) -> Self {
        self.moe_enabled = true;
        self.moe_num_experts = num_experts;
        self.moe_top_k = top_k;
        self
    }

    /// Builder: enable gradient checkpointing with N segments
    pub fn with_gradient_checkpointing(mut self, segments: usize) -> Self {
        self.gradient_checkpoint_segments = segments;
        self
    }

    /// Calculate total parameters (approximate)
    pub fn total_params(&self) -> usize {
        let embedding_params = self.vocab_size * self.hidden_size;

        let attention_params = self.num_layers * (
            // QKV projections (with FlyLoRA reduction)
            3 * self.hidden_size * (self.flylora_rank * 2) +
            // Output projection
            self.hidden_size * self.hidden_size
        );

        let ffn_params_per_expert = 2 * self.hidden_size * self.intermediate_size;
        let ffn_params = self.num_layers * if self.moe_enabled {
            // Router: hidden_size -> num_experts
            let router_params = self.hidden_size * self.moe_num_experts;
            router_params + self.moe_num_experts * ffn_params_per_expert
        } else {
            ffn_params_per_expert
        };

        let norm_params = self.num_layers * 2 * self.hidden_size;

        embedding_params + attention_params + ffn_params + norm_params
    }

    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.hidden_size % self.num_heads != 0 {
            anyhow::bail!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size,
                self.num_heads
            );
        }

        if self.flylora_sparsity < 0.0 || self.flylora_sparsity > 1.0 {
            anyhow::bail!("flylora_sparsity must be between 0 and 1");
        }

        if self.moe_enabled {
            if self.moe_top_k == 0 || self.moe_top_k > self.moe_num_experts {
                anyhow::bail!(
                    "moe_top_k ({}) must be between 1 and moe_num_experts ({})",
                    self.moe_top_k,
                    self.moe_num_experts
                );
            }
            if self.moe_num_experts < 2 {
                anyhow::bail!("moe_num_experts must be at least 2");
            }
        }

        if self.gradient_checkpoint_segments > self.num_layers {
            anyhow::bail!(
                "gradient_checkpoint_segments ({}) must be <= num_layers ({})",
                self.gradient_checkpoint_segments,
                self.num_layers
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_sizes() {
        let tiny = VesperConfig::tiny();
        let small = VesperConfig::small();
        let medium = VesperConfig::medium();
        let large = VesperConfig::large();

        assert!(tiny.total_params() < small.total_params());
        assert!(small.total_params() < medium.total_params());
        assert!(medium.total_params() < large.total_params());
    }

    #[test]
    fn test_validation() {
        let valid = VesperConfig::default();
        assert!(valid.validate().is_ok());

        let mut invalid = VesperConfig::default();
        invalid.hidden_size = 777; // Not divisible by num_heads
        assert!(invalid.validate().is_err());
    }
}
