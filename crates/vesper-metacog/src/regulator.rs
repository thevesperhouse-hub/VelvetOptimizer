//! Metacognitive Regulator - Three-stage regulation process

use candle_core::Result;
use serde::{Deserialize, Serialize};

use crate::meta_head::{MetaHead, MetaOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatorConfig {
    pub confidence_threshold: f64,
    pub max_iterations: usize,
    pub enable_planning: bool,
    pub enable_regulation: bool,
    pub enable_termination: bool,
}

impl Default for RegulatorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.85,
            max_iterations: 3,
            enable_planning: true,
            enable_regulation: true,
            enable_termination: true,
        }
    }
}

pub struct MetacognitiveRegulator {
    config: RegulatorConfig,
    meta_head: MetaHead,
}

impl MetacognitiveRegulator {
    pub fn new(config: RegulatorConfig, meta_head: MetaHead) -> Self {
        Self { config, meta_head }
    }

    /// Three-stage metacognitive process
    pub fn regulate(&self, meta_output: &MetaOutput, iteration: usize) -> Result<RegulationDecision> {
        // Stage 1: Proactive Planning (via CASCADE - not implemented here)
        if self.config.enable_planning && iteration == 0 {
            // Planning would happen at the model level
        }

        // Stage 2: Online Regulation - Error detection
        let has_errors = if self.config.enable_regulation {
            self.detect_errors(meta_output)?
        } else {
            false
        };

        // Stage 3: Satisficing Termination
        let should_terminate = if self.config.enable_termination {
            self.should_terminate(meta_output, iteration, has_errors)?
        } else {
            false
        };

        Ok(RegulationDecision {
            should_terminate,
            has_errors,
            iteration,
            confidence: self.extract_confidence(meta_output)?,
        })
    }

    fn detect_errors(&self, meta_output: &MetaOutput) -> Result<bool> {
        let error_probs = candle_nn::ops::softmax(&meta_output.error_logits, 2)?;
        let max_error_prob = error_probs.max(2)?.mean_all()?.to_scalar::<f32>()?;
        
        Ok(max_error_prob > 0.5)
    }

    fn should_terminate(
        &self,
        meta_output: &MetaOutput,
        iteration: usize,
        has_errors: bool,
    ) -> Result<bool> {
        // Force stop if max iterations reached
        if iteration >= self.config.max_iterations {
            return Ok(true);
        }

        // Check if satisficing criteria met
        let confidence = self.extract_confidence(meta_output)?;
        let is_confident = confidence > self.config.confidence_threshold;

        // Terminate if confident and no errors
        Ok(is_confident && !has_errors)
    }

    fn extract_confidence(&self, meta_output: &MetaOutput) -> Result<f64> {
        let confidence = meta_output.confidence.mean_all()?.to_scalar::<f32>()?;
        Ok(confidence as f64)
    }

    pub fn config(&self) -> &RegulatorConfig {
        &self.config
    }
}

#[derive(Debug)]
pub struct RegulationDecision {
    pub should_terminate: bool,
    pub has_errors: bool,
    pub iteration: usize,
    pub confidence: f64,
}

impl RegulationDecision {
    pub fn should_continue(&self) -> bool {
        !self.should_terminate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_head::{MetaHead, MetaHeadConfig};
    use candle_core::{Device, DType, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_regulator() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let meta_head = MetaHead::new(MetaHeadConfig::default(), vb)?;
        let regulator = MetacognitiveRegulator::new(RegulatorConfig::default(), meta_head);
        
        let hidden = Tensor::randn(0f32, 1.0, (2, 32, 768), &device)?;
        let meta_output = regulator.meta_head.forward(&hidden)?;
        
        let decision = regulator.regulate(&meta_output, 0)?;
        
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert_eq!(decision.iteration, 0);
        
        Ok(())
    }
}
