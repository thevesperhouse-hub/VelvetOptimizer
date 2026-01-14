//! Auto-Scaling using Chinchilla Laws

use anyhow::Result;
use serde::{Deserialize, Serialize};
use vesper_core::VesperConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingResult {
    pub config: VesperConfig,
    pub dataset_tokens: usize,
    pub optimal_params: usize,
    pub recommended_epochs: usize,
    pub overtraining_factor: f64,
}

pub struct AutoScaler {
    chinchilla_ratio: f64, // tokens per parameter (default: 20)
}

impl Default for AutoScaler {
    fn default() -> Self {
        Self {
            chinchilla_ratio: 20.0,
        }
    }
}

impl AutoScaler {
    pub fn new(chinchilla_ratio: f64) -> Self {
        Self { chinchilla_ratio }
    }

    /// Calculate optimal model size for given dataset
    pub fn calculate_optimal_size(&self, dataset_tokens: usize) -> usize {
        (dataset_tokens as f64 / self.chinchilla_ratio) as usize
    }

    /// Generate config based on target parameter count
    pub fn generate_config(&self, target_params: usize) -> Result<VesperConfig> {
        // Heuristic to find hidden_size and num_layers
        // Formula: params ≈ vocab_size * hidden_size + num_layers * (4 * hidden_size^2)
        
        let vocab_size = 32000;
        
        // Try different hidden sizes
        for hidden_size in (256..=2048).step_by(128) {
            for num_layers in 6..=32 {
                let config = VesperConfig {
                    vocab_size,
                    hidden_size,
                    num_layers,
                    num_heads: (hidden_size / 64).max(4),
                    intermediate_size: hidden_size * 4,
                    ..Default::default()
                };

                let actual_params = config.total_params();
                
                // Accept if within 10% of target
                let diff_ratio = (actual_params as f64 - target_params as f64).abs() 
                    / target_params as f64;
                
                if diff_ratio < 0.1 {
                    return Ok(config);
                }
            }
        }

        // Fallback: use medium config
        Ok(VesperConfig::medium())
    }

    /// Calculate recommended epochs based on Chinchilla ratio
    pub fn calculate_epochs(
        &self,
        dataset_tokens: usize,
        model_params: usize,
    ) -> usize {
        let optimal_tokens = model_params as f64 * self.chinchilla_ratio;
        let overtraining_factor = optimal_tokens / dataset_tokens as f64;
        
        // Base epochs: 3
        // Multiply by overtraining factor (clamped to 1-20)
        let epochs = (3.0 * overtraining_factor.max(1.0).min(20.0)) as usize;
        
        epochs.max(1)
    }

    /// Complete scaling analysis
    pub fn scale(&self, dataset_tokens: usize) -> Result<ScalingResult> {
        let optimal_params = self.calculate_optimal_size(dataset_tokens);
        let config = self.generate_config(optimal_params)?;
        let actual_params = config.total_params();
        
        let recommended_epochs = self.calculate_epochs(dataset_tokens, actual_params);
        let overtraining_factor = (actual_params as f64 * self.chinchilla_ratio) 
            / dataset_tokens as f64;

        Ok(ScalingResult {
            config,
            dataset_tokens,
            optimal_params,
            recommended_epochs,
            overtraining_factor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_size() {
        let scaler = AutoScaler::default();
        
        // 1M tokens -> 50K params
        assert_eq!(scaler.calculate_optimal_size(1_000_000), 50_000);
        
        // 100M tokens -> 5M params
        assert_eq!(scaler.calculate_optimal_size(100_000_000), 5_000_000);
    }

    #[test]
    fn test_config_generation() -> Result<()> {
        let scaler = AutoScaler::default();
        
        let config = scaler.generate_config(50_000_000)?;
        let params = config.total_params();
        
        // Should be within 10% of target
        let diff_ratio = (params as f64 - 50_000_000.0).abs() / 50_000_000.0;
        assert!(diff_ratio < 0.1);
        
        Ok(())
    }

    #[test]
    fn test_epochs_calculation() {
        let scaler = AutoScaler::default();
        
        // Perfect Chinchilla ratio: 3 epochs
        let epochs = scaler.calculate_epochs(1_000_000, 50_000);
        assert_eq!(epochs, 3);
        
        // Underdata (need more epochs)
        let epochs = scaler.calculate_epochs(500_000, 50_000);
        assert!(epochs > 3);
    }
}
