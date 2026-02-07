//! ERA Activation - Entropy-Regularized Activation
//!
//! ERA(x) = GELU(x) + γ * softplus(x)
//! where softplus(x) = log(1 + exp(x))
//!
//! The additive softplus term prevents dead neurons by ensuring
//! a small positive baseline activation, acting as implicit
//! entropy regularization on the hidden representations.

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ERAConfig {
    pub gamma: f64,
}

impl Default for ERAConfig {
    fn default() -> Self {
        Self {
            gamma: 0.1,
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

    /// Forward pass: ERA(x) = GELU(x) + γ * softplus(x)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gelu_x = gelu(x)?;

        if self.config.gamma > 0.0 {
            let sp = softplus(x)?;
            gelu_x + (sp * self.config.gamma)?
        } else {
            Ok(gelu_x)
        }
    }

    pub fn config(&self) -> &ERAConfig {
        &self.config
    }
}

/// GELU activation (tanh approximation):
/// 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = (2.0_f64 / std::f64::consts::PI).sqrt();
    let x_cubed = x.mul(&x.mul(x)?)?;
    let inner = ((x + (x_cubed * 0.044715)?)? * coeff)?;
    let tanh_inner = inner.tanh()?;
    (x * 0.5)?.mul(&(tanh_inner + 1.0)?)
}

/// Softplus: log(1 + exp(x))
fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0)?.log()
}

/// Simplified ERA for inference (just GELU, no entropy regularization)
pub fn era_simple(x: &Tensor) -> Result<Tensor> {
    gelu(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
    fn test_era_greater_than_gelu() -> Result<()> {
        let device = Device::Cpu;
        let era = ERAActivation::new(ERAConfig { gamma: 0.1 });

        // For positive inputs, ERA > GELU (since softplus > 0)
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        let era_out = era.forward(&x)?.to_vec1::<f32>()?;
        let gelu_out = gelu(&x)?.to_vec1::<f32>()?;

        for (e, g) in era_out.iter().zip(gelu_out.iter()) {
            assert!(e > g, "ERA({}) should be > GELU({})", e, g);
        }

        Ok(())
    }
}
