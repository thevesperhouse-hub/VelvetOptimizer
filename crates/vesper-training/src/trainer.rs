//! Training orchestration

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::loss::cross_entropy;
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
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            num_epochs: 3,
            batch_size: 4,
            learning_rate: 3e-4,
            log_interval: 10,
            save_interval: 100,
            output_dir: "checkpoints".to_string(),
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
        optimizer: &mut VelvetOptimizer,
        dataset: &mut Dataset,
    ) -> Result<TrainingMetrics> {
        println!("\n🏋️  Starting training...");
        println!("   Epochs: {}", self.config.num_epochs);
        println!("   Batch size: {}", self.config.batch_size);
        println!("   Learning rate: {:.1e}", self.config.learning_rate);

        let mut metrics = TrainingMetrics::new();

        for epoch in 0..self.config.num_epochs {
            println!("\n[Epoch {}/{}]", epoch + 1, self.config.num_epochs);
            
            // Shuffle dataset at the start of each epoch
            dataset.shuffle();
            
            let epoch_metrics = self.train_epoch(model, optimizer, dataset)?;
            metrics.epochs.push(epoch_metrics);

            println!("  ✓ Epoch {} complete - avg loss: {:.4}", 
                epoch + 1, 
                metrics.epochs[epoch].avg_loss
            );
        }

        Ok(metrics)
    }

    fn train_epoch(
        &self,
        model: &VesperLM,
        _optimizer: &mut VelvetOptimizer,
        dataset: &Dataset,
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

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * self.config.batch_size;
            let end_idx = (start_idx + self.config.batch_size).min(dataset.len());

            // Get batch
            let batch = self.prepare_batch(dataset, start_idx, end_idx)?;

            // Forward pass
            let logits = model.forward(&batch.input_ids, Some(&batch.attention_mask))?;

            // Compute loss (cross-entropy for language modeling)
            let loss = self.compute_loss(&logits, &batch.labels)?;
            let loss_val = loss.to_scalar::<f32>()?;

            total_loss += loss_val;
            num_steps += 1;

            // TODO: Backward pass
            // TODO: Optimizer step

            if batch_idx % self.config.log_interval == 0 {
                pb.set_message(format!("loss: {:.4}", loss_val));
            }

            pb.inc(1);
        }

        pb.finish_with_message("done");

        Ok(EpochMetrics {
            avg_loss: total_loss / num_steps as f32,
            num_steps,
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

        // Convert to tensors
        let _batch_size = input_ids_batch.len();
        let _seq_len = input_ids_batch[0].len();

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
}
