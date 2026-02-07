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

    /// Builder: set vocab_size (must match tokenizer)
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
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
        
        let ffn_params = self.num_layers * (
            2 * self.hidden_size * self.intermediate_size
        );
        
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
