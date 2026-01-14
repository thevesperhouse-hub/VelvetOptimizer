//! ERA Activation - Entropy-Regularized Activation
//! 
//! Custom activation combining SiLU with entropy regularization

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ERAConfig {
    pub temperature: f64,
    pub entropy_weight: f64,
}

impl Default for ERAConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            entropy_weight: 0.01,
        }
    }
}

pub struct ERAActivation {
    config: ERAConfig,
}

impl ERAActivation {
    pub fn new(config: ERAConfig) -> Self {
        Self { config }
    }

    /// Forward pass: ERA(x) = x * sigmoid(x) * (1 - entropy_penalty)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SiLU/Swish: x * sigmoid(x)
        let sigmoid_x = candle_nn::ops::sigmoid(x)?;
        let silu = x.mul(&sigmoid_x)?;

        // Calculate entropy penalty if weight > 0
        if self.config.entropy_weight > 0.0 {
            let entropy_penalty = self.calculate_entropy_penalty(x)?;
            let one_minus_penalty = (1.0 - entropy_penalty)?;
            silu.broadcast_mul(&one_minus_penalty)
        } else {
            Ok(silu)
        }
    }

    /// Calculate entropy-based regularization
    fn calculate_entropy_penalty(&self, x: &Tensor) -> Result<Tensor> {
        // Softmax over last dimension
        let temp_scaled = (x / self.config.temperature)?;
        let probs = candle_nn::ops::softmax(&temp_scaled, x.dims().len() - 1)?;

        // Entropy: -sum(p * log(p)) with epsilon to avoid log(0)
        let eps = 1e-10;
        let probs_safe = (probs.clone() + eps)?;
        let log_probs = probs_safe.log()?;
        let entropy = (probs.mul(&log_probs)?.sum_keepdim(x.dims().len() - 1)? * -1.0)?;

        // Normalize to [0, 1] range
        let max_entropy = (x.dims()[x.dims().len() - 1] as f64).ln();
        let normalized_entropy = (entropy / max_entropy)?;

        // Scale by weight and clamp to avoid NaN
        let penalty = (normalized_entropy * self.config.entropy_weight)?;
        penalty.clamp(0.0, 1.0)
    }

    pub fn config(&self) -> &ERAConfig {
        &self.config
    }
}

/// Simplified ERA for inference (no entropy penalty)
pub fn era_simple(x: &Tensor) -> Result<Tensor> {
    let sigmoid_x = candle_nn::ops::sigmoid(x)?;
    x.mul(&sigmoid_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_era_forward() -> Result<()> {
        let device = Device::Cpu;
        let era = ERAActivation::new(ERAConfig::default());

        let x = Tensor::randn(0f32, 1.0, (2, 4, 768), &device)?;
        let output = era.forward(&x)?;

        assert_eq!(output.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_era_simple() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 4, 768), &device)?;
        let output = era_simple(&x)?;

        assert_eq!(output.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_entropy_penalty() -> Result<()> {
        let device = Device::Cpu;
        let era = ERAActivation::new(ERAConfig {
            temperature: 0.1,
            entropy_weight: 0.1,
        });

        let x = Tensor::randn(0f32, 1.0, (2, 4, 768), &device)?;
        let penalty = era.calculate_entropy_penalty(&x)?;

        // Penalty should be between 0 and entropy_weight
        let max_val = penalty.max(0)?.to_vec0::<f32>()?;
        assert!(max_val <= 0.1);

        Ok(())
    }
}
