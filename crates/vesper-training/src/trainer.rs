//! Training orchestration

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{self, loss::cross_entropy, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use vesper_core::VesperLM;
use vesper_optimizer::VelvetOptimizer;
use crate::dataset::Dataset;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub log_interval: usize,
    pub save_interval: usize,
    pub output_dir: String,
    pub max_steps: Option<usize>,
    pub warmup_steps: usize,
    pub gradient_accumulation_steps: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            num_epochs: 3,
            batch_size: 4,
            learning_rate: 3e-4,
            log_interval: 10,
            save_interval: 500,
            output_dir: "checkpoints".to_string(),
            max_steps: None,
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
        }
    }
}

pub struct Trainer {
    config: TrainerConfig,
    device: Device,
}

impl Trainer {
    pub fn new(config: TrainerConfig, device: Device) -> Self {
        Self { config, device }
    }

    pub fn train(
        &self,
        model: &VesperLM,
        varmap: &VarMap,
        optimizer: &mut VelvetOptimizer,
        dataset: &mut Dataset,
    ) -> Result<TrainingMetrics> {
        println!("\n  Starting training...");
        println!("   Epochs: {}", self.config.num_epochs);
        println!("   Batch size: {}", self.config.batch_size);
        println!("   Learning rate: {:.1e}", self.config.learning_rate);
        println!("   Gradient accumulation: {}", self.config.gradient_accumulation_steps);
        if let Some(max) = self.config.max_steps {
            println!("   Max steps: {}", max);
        }

        // Create output directory for checkpoints
        std::fs::create_dir_all(&self.config.output_dir)?;

        let vocab_size = model.config().vocab_size;

        let mut metrics = TrainingMetrics::new();
        let mut global_step: usize = 0;

        for epoch in 0..self.config.num_epochs {
            println!("\n[Epoch {}/{}]", epoch + 1, self.config.num_epochs);

            // Shuffle dataset at the start of each epoch
            dataset.shuffle();

            let epoch_metrics = self.train_epoch(model, varmap, optimizer, dataset, &mut global_step, vocab_size)?;
            metrics.epochs.push(epoch_metrics);

            println!("  Epoch {} complete - avg loss: {:.4}",
                epoch + 1,
                metrics.epochs[epoch].avg_loss
            );

            // Check max_steps
            if let Some(max) = self.config.max_steps {
                if global_step >= max {
                    println!("  Reached max_steps ({}), stopping.", max);
                    break;
                }
            }
        }

        // Save final checkpoint
        let final_path = format!("{}/checkpoint-final.safetensors", self.config.output_dir);
        self.save_checkpoint(varmap, &final_path)?;
        println!("\n  Final checkpoint saved to: {}", final_path);

        Ok(metrics)
    }

    fn train_epoch(
        &self,
        model: &VesperLM,
        varmap: &VarMap,
        optimizer: &mut VelvetOptimizer,
        dataset: &Dataset,
        global_step: &mut usize,
        vocab_size: usize,
    ) -> Result<EpochMetrics> {
        let num_batches = (dataset.len() + self.config.batch_size - 1) / self.config.batch_size;

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}")
                .unwrap()
        );

        let mut total_loss = 0.0;
        let mut num_steps = 0;
        let mut accum_count = 0;
        let mut step_losses = Vec::new();

        for batch_idx in 0..num_batches {
            // Check max_steps
            if let Some(max) = self.config.max_steps {
                if *global_step >= max {
                    break;
                }
            }

            let start_idx = batch_idx * self.config.batch_size;
            let end_idx = (start_idx + self.config.batch_size).min(dataset.len());

            // Get batch
            let batch = self.prepare_batch(dataset, start_idx, end_idx)?;

            // Forward pass
            let (logits, _aux_loss) = model.forward(&batch.input_ids, Some(&batch.attention_mask))?;

            // Compute loss (cross-entropy for language modeling)
            let loss = self.compute_loss(&logits, &batch.labels)?;
            let loss_val = loss.to_scalar::<f32>()?;

            total_loss += loss_val;
            num_steps += 1;
            accum_count += 1;
            step_losses.push(loss_val);

            // Backward pass + optimizer step (with gradient accumulation)
            if accum_count >= self.config.gradient_accumulation_steps {
                // 1. Entropy-Adaptive LR: real logits entropy
                let current_entropy = compute_logits_entropy(&logits, vocab_size)?;
                let baseline_entropy = 0.5;
                let entropy_scale = (current_entropy / baseline_entropy).clamp(0.5, 2.0);
                optimizer.set_entropy_scale(entropy_scale);

                // 2. Perplexity-Guided Momentum: baseline_ppl=40.0
                let current_ppl = (loss_val as f64).exp();
                let baseline_ppl = 40.0;
                let ppl_scale = (baseline_ppl / current_ppl.max(1.0)).clamp(0.5, 2.0);
                optimizer.set_perplexity_scale(ppl_scale);

                // Single-call backward + step (like AdamW's backward_step)
                optimizer.backward_step(&loss, varmap)?;

                accum_count = 0;
                *global_step += 1;

                // Checkpoint saving
                if self.config.save_interval > 0
                    && *global_step % self.config.save_interval == 0
                    && *global_step > 0
                {
                    let path = format!("{}/checkpoint-{}.safetensors",
                        self.config.output_dir, global_step);
                    self.save_checkpoint(varmap, &path)?;
                    pb.println(format!("  Checkpoint saved: {}", path));
                }
            }

            if batch_idx % self.config.log_interval == 0 {
                pb.set_message(format!("loss: {:.4} | step: {}", loss_val, global_step));
            }

            pb.inc(1);
        }

        pb.finish_with_message("done");

        Ok(EpochMetrics {
            avg_loss: total_loss / num_steps.max(1) as f32,
            num_steps,
            step_losses,
        })
    }

    fn prepare_batch(
        &self,
        dataset: &Dataset,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<Batch> {
        let mut input_ids_batch = Vec::new();
        let mut attention_mask_batch = Vec::new();
        let mut labels_batch = Vec::new();

        for idx in start_idx..end_idx {
            if let Some(sample) = dataset.get(idx) {
                input_ids_batch.push(sample.input_ids.clone());
                attention_mask_batch.push(sample.attention_mask.clone());
                labels_batch.push(sample.labels.clone());
            }
        }

        if input_ids_batch.is_empty() {
            anyhow::bail!("Empty batch at indices {}..{}", start_idx, end_idx);
        }

        // Convert to tensors
        let input_ids = Tensor::new(input_ids_batch, &self.device)?;
        let attention_mask = Tensor::new(attention_mask_batch, &self.device)?;
        let labels = Tensor::new(labels_batch, &self.device)?;

        Ok(Batch {
            input_ids,
            attention_mask,
            labels,
        })
    }

    fn compute_loss(&self, logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        // logits: [batch_size, seq_len, vocab_size]
        // labels: [batch_size, seq_len]

        let (batch_size, seq_len, vocab_size) = logits.dims3()?;

        // Reshape for cross_entropy
        let logits_2d = logits.reshape((batch_size * seq_len, vocab_size))?;
        let labels_1d = labels.flatten_all()?;

        cross_entropy(&logits_2d, &labels_1d)
            .map_err(|e| anyhow::anyhow!("Cross entropy failed: {}", e))
    }

    fn save_checkpoint(&self, varmap: &VarMap, path: &str) -> Result<()> {
        varmap.save(path)?;
        Ok(())
    }
}

/// Compute normalized entropy from logits (matches Python reference).
/// Returns mean(entropy / log(vocab_size)), in [0, 1] range.
fn compute_logits_entropy(logits: &Tensor, vocab_size: usize) -> Result<f64> {
    let max_entropy = (vocab_size as f64).ln();
    let last_dim = logits.dims().len() - 1;

    // Detach to avoid tracking in computation graph
    let logits_d = logits.detach();

    // Softmax probabilities
    let probs = candle_nn::ops::softmax(&logits_d, last_dim)?;

    // log(probs) with numerical safety
    let log_probs = probs.clamp(1e-10, 1.0)?.log()?;

    // entropy = -sum(p * log(p), dim=-1)
    let entropy = (probs.mul(&log_probs)?.sum(last_dim)? * -1.0)?;

    // Normalize by max entropy -> [0, 1]
    let normalized = (entropy / max_entropy)?;

    // Mean over all positions
    let mean = normalized.mean_all()?.to_scalar::<f32>()? as f64;

    Ok(mean)
}

struct Batch {
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epochs: Vec<EpochMetrics>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self { epochs: Vec::new() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub avg_loss: f32,
    pub num_steps: usize,
    pub step_losses: Vec<f32>,
}
