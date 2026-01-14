//! Velvet Optimizer Implementation
//! 
//! High-performance optimizer with adaptive features
//! Utilise les kernels CUDA custom quand disponible

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use crate::cuda::velvet_update_cuda;

#[derive(Debug, Clone)]
pub struct VelvetConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub entropy_adaptive: bool,
    pub perplexity_guided: bool,
    pub sparse_aware: bool,
}

impl Default for VelvetConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            entropy_adaptive: false,
            perplexity_guided: false,
            sparse_aware: false,
        }
    }
}

impl VelvetConfig {
    /// Optimal config from benchmarks
    pub fn optimal() -> Self {
        Self {
            lr: 5e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-3,
            entropy_adaptive: true,
            perplexity_guided: true,
            sparse_aware: true,
        }
    }
}

pub struct VelvetOptimizer {
    config: VelvetConfig,
    step: usize,
    entropy_scale: f64,
    perplexity_scale: f64,
    
    // State for each parameter
    state: HashMap<String, OptimizerState>,
}

struct OptimizerState {
    m: Tensor,  // First moment
    v: Tensor,  // Second moment
    step: usize,
}

impl VelvetOptimizer {
    pub fn new(config: VelvetConfig) -> Self {
        Self {
            config,
            step: 0,
            entropy_scale: 1.0,
            perplexity_scale: 1.0,
            state: HashMap::new(),
        }
    }

    /// Step with explicit gradients (Candle doesn't have .grad() on Tensor)
    pub fn step(&mut self, params: &mut [(String, Tensor)], grads: &[(String, Tensor)]) -> Result<()> {
        self.step += 1;

        for (name, grad) in grads.iter() {
            if let Some((_, param)) = params.iter_mut().find(|(n, _)| n == name) {
                self.update_param(name, param, grad)?;
            }
        }

        Ok(())
    }

    /// Simple step for single param/grad pair
    pub fn step_single(&mut self, name: &str, param: &mut Tensor, grad: &Tensor) -> Result<()> {
        self.step += 1;
        self.update_param(name, param, grad)
    }

    fn update_param(&mut self, name: &str, param: &mut Tensor, grad: &Tensor) -> Result<()> {
        // Get or create state
        if !self.state.contains_key(name) {
            self.state.insert(name.to_string(), OptimizerState {
                m: Tensor::zeros_like(param)?,
                v: Tensor::zeros_like(param)?,
                step: 0,
            });
        }

        let state = self.state.get_mut(name).unwrap();
        state.step += 1;

        // Bias corrections
        let bias_correction1 = 1.0 - self.config.beta1.powi(state.step as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(state.step as i32);

        // Adaptive LR
        let effective_lr = if self.config.entropy_adaptive {
            self.config.lr * self.entropy_scale
        } else {
            self.config.lr
        };

        // Adaptive momentum
        let effective_beta1 = if self.config.perplexity_guided {
            (self.config.beta1 * self.perplexity_scale).clamp(0.5, 0.999)
        } else {
            self.config.beta1
        };

        // ===== CUDA KERNEL PATH =====
        #[cfg(feature = "cuda")]
        if matches!(param.device(), Device::Cuda(_)) {
            let state = self.state.get(name).unwrap();
            velvet_update_cuda(
                param,
                &state.m,
                &state.v,
                grad,
                effective_lr as f32,
                effective_beta1 as f32,
                self.config.beta2 as f32,
                self.config.eps as f32,
                self.config.weight_decay as f32,
                bias_correction1 as f32,
                bias_correction2 as f32,
                self.config.entropy_adaptive,
                self.entropy_scale as f32,
                self.config.perplexity_guided,
                self.perplexity_scale as f32,
                self.config.sparse_aware,
            )?;
            return Ok(());
        }

        // ===== CPU FALLBACK (Candle ops) =====
        let config_beta2 = self.config.beta2;
        let config_eps = self.config.eps;
        let config_wd = self.config.weight_decay;

        // Decoupled weight decay
        *param = (param.clone() * (1.0 - effective_lr * config_wd))?;

        // Update moments
        let state = self.state.get_mut(name).unwrap();
        state.m = (state.m.clone() * effective_beta1)?.add(&(grad * (1.0 - effective_beta1))?)?;
        state.v = (state.v.clone() * config_beta2)?.add(&(grad.sqr()? * (1.0 - config_beta2))?)?;

        // Bias-corrected moments
        let m_hat = (state.m.clone() / bias_correction1)?;
        let v_hat = (state.v.clone() / bias_correction2)?;

        // Parameter update
        let update = (m_hat / (v_hat.sqrt()? + config_eps)?)?;
        *param = (param.clone() - (update * effective_lr)?)?;

        Ok(())
    }

    pub fn set_entropy_scale(&mut self, scale: f64) {
        self.entropy_scale = scale;
    }

    pub fn set_perplexity_scale(&mut self, scale: f64) {
        self.perplexity_scale = scale;
    }

    pub fn get_step(&self) -> usize {
        self.step
    }

    pub fn get_lr(&self) -> f64 {
        self.config.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_optimizer_creation() {
        let config = VelvetConfig::default();
        let optimizer = VelvetOptimizer::new(config);
        assert_eq!(optimizer.get_step(), 0);
    }

    #[test]
    fn test_optimal_config() {
        let config = VelvetConfig::optimal();
        assert_eq!(config.lr, 5e-4);
        assert_eq!(config.weight_decay, 1e-3);
        assert!(config.entropy_adaptive);
        assert!(config.perplexity_guided);
        assert!(config.sparse_aware);
    }
}
