//! MetaHead - Error detection and confidence estimation

use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaHeadConfig {
    pub hidden_size: usize,
    pub meta_hidden: usize,
    pub num_error_types: usize,
}

impl Default for MetaHeadConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            meta_hidden: 256,
            num_error_types: 3, // Factual, Logical, Incomplete
        }
    }
}

pub struct MetaHead {
    config: MetaHeadConfig,
    
    // Error detection
    error_proj: Linear,
    error_classifier: Linear,
    
    // Confidence estimation
    confidence_proj: Linear,
    
    // Termination decision
    termination_proj: Linear,
}

impl MetaHead {
    pub fn new(config: MetaHeadConfig, vb: VarBuilder) -> Result<Self> {
        let error_proj = candle_nn::linear(
            config.hidden_size,
            config.meta_hidden,
            vb.pp("error_proj"),
        )?;
        
        let error_classifier = candle_nn::linear(
            config.meta_hidden,
            config.num_error_types,
            vb.pp("error_classifier"),
        )?;
        
        let confidence_proj = candle_nn::linear(
            config.hidden_size,
            1,
            vb.pp("confidence_proj"),
        )?;
        
        let termination_proj = candle_nn::linear(
            config.hidden_size,
            1,
            vb.pp("termination_proj"),
        )?;

        Ok(Self {
            config,
            error_proj,
            error_classifier,
            confidence_proj,
            termination_proj,
        })
    }

    /// Forward pass: detect errors, estimate confidence, decide termination
    pub fn forward(&self, hidden_states: &Tensor) -> Result<MetaOutput> {
        // Error detection
        let error_features = self.error_proj.forward(hidden_states)?;
        let error_features = error_features.gelu()?;
        let error_logits = self.error_classifier.forward(&error_features)?;
        
        // Confidence estimation (sigmoid for [0, 1])
        let confidence_logits = self.confidence_proj.forward(hidden_states)?;
        let confidence = candle_nn::ops::sigmoid(&confidence_logits)?;
        
        // Termination decision (sigmoid for probability)
        let termination_logits = self.termination_proj.forward(hidden_states)?;
        let should_terminate = candle_nn::ops::sigmoid(&termination_logits)?;

        Ok(MetaOutput {
            error_logits,
            confidence,
            should_terminate,
        })
    }

    pub fn config(&self) -> &MetaHeadConfig {
        &self.config
    }
}

#[derive(Debug)]
pub struct MetaOutput {
    pub error_logits: Tensor,      // [batch, seq, num_error_types]
    pub confidence: Tensor,         // [batch, seq, 1]
    pub should_terminate: Tensor,   // [batch, seq, 1]
}

impl MetaOutput {
    /// Check if confident and no errors detected
    pub fn is_satisficing(&self, threshold: f64) -> Result<bool> {
        // Average confidence across sequence
        let avg_confidence = self.confidence.mean_all()?.to_scalar::<f32>()?;
        
        // Check if any error is predicted
        let error_probs = candle_nn::ops::softmax(&self.error_logits, 2)?;
        let max_error_prob = error_probs.max(2)?.mean_all()?.to_scalar::<f32>()?;
        
        Ok(avg_confidence as f64 > threshold && max_error_prob < 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_meta_head() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let config = MetaHeadConfig::default();
        let meta_head = MetaHead::new(config, vb)?;
        
        let hidden = Tensor::randn(0f32, 1.0, (2, 32, 768), &device)?;
        let output = meta_head.forward(&hidden)?;
        
        assert_eq!(output.error_logits.dims(), &[2, 32, 3]);
        assert_eq!(output.confidence.dims(), &[2, 32, 1]);
        assert_eq!(output.should_terminate.dims(), &[2, 32, 1]);
        
        Ok(())
    }
}
