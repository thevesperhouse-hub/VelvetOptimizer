//! Benchmark subcommand - Compare Velvet vs AdamW

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{self, loss::cross_entropy, VarMap, VarBuilder, Optimizer, optim::AdamW, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

use vesper_core::{VesperConfig, VesperLM};
use vesper_optimizer::{VelvetOptimizer, VelvetConfig};
use vesper_training::DatasetLoader;

use crate::tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepMetric {
    step: usize,
    loss: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    entropy_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ppl_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    grad_norm: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    optimizer: String,
    model_size: String,
    epochs: usize,
    final_loss: f32,
    best_loss: f32,
    training_time_secs: f64,
    loss_history: Vec<f32>,
    perplexity_history: Vec<f32>,
    step_metrics: Vec<StepMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkReport {
    velvet: BenchmarkResult,
    adamw: BenchmarkResult,
    comparison: Comparison,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Comparison {
    loss_improvement_pct: f64,
    perplexity_improvement_pct: f64,
    speed_difference_pct: f64,
    velvet_final_loss: f32,
    adamw_final_loss: f32,
    velvet_final_ppl: f32,
    adamw_final_ppl: f32,
}

/// Run the benchmark subcommand
pub fn run(
    dataset_path: PathBuf,
    tokenizer_name: String,
    model_size: String,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    seq_len: usize,
    output: PathBuf,
) -> Result<()> {
    println!("\n=== VesperAI Benchmark: Velvet vs AdamW ===\n");

    let device = if candle_core::utils::cuda_is_available() {
        println!("  CUDA available, using GPU");
        Device::new_cuda(0)?
    } else {
        println!("  No CUDA, using CPU");
        Device::Cpu
    };

    // Load tokenizer and dataset
    let tok = tokenizer::load_tokenizer(&tokenizer_name)?;
    let vocab_size = tokenizer::vocab_size(&tok);

    let loader = DatasetLoader::from_tokenizer(tok, seq_len);
    let format = match dataset_path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => "jsonl",
        Some("json") => "squad",
        _ => "text",
    };

    println!("  Loading dataset...");
    let dataset = match format {
        "jsonl" => loader.load_jsonl(&dataset_path)?,
        "squad" | "json" => loader.load_squad(&dataset_path)?,
        _ => loader.load_text(&dataset_path)?,
    };
    println!("  Dataset: {} samples", dataset.len());

    let config = match model_size.as_str() {
        "tiny" => VesperConfig::tiny(),
        "small" => VesperConfig::small(),
        "medium" => VesperConfig::medium(),
        "large" => VesperConfig::large(),
        "xlarge" | "1b" => VesperConfig::xlarge(),
        _ => anyhow::bail!("Unknown model size: {}", model_size),
    };
    let config = config.with_vocab_size(vocab_size);
    config.validate()?;

    println!("  Model: {} ({} params)", model_size, config.total_params());
    println!("  Epochs: {}, Batch size: {}, LR: {:.1e}", epochs, batch_size, lr);

    // === Run Velvet first ===
    println!("\n--- Velvet Training ---");
    let velvet_result = run_velvet(
        &config, &dataset, &device,
        epochs, batch_size, lr, seq_len,
    )?;

    // === Run AdamW ===
    println!("\n--- AdamW Training ---");
    let adamw_result = run_adamw(
        &config, &dataset, &device,
        epochs, batch_size, lr, seq_len,
    )?;

    // === Compare ===
    let comparison = Comparison {
        loss_improvement_pct: ((adamw_result.final_loss - velvet_result.final_loss)
            / adamw_result.final_loss as f32 * 100.0) as f64,
        perplexity_improvement_pct: {
            let v_ppl = velvet_result.perplexity_history.last().copied().unwrap_or(0.0);
            let a_ppl = adamw_result.perplexity_history.last().copied().unwrap_or(0.0);
            ((a_ppl - v_ppl) / a_ppl * 100.0) as f64
        },
        speed_difference_pct: ((adamw_result.training_time_secs - velvet_result.training_time_secs)
            / adamw_result.training_time_secs * 100.0),
        velvet_final_loss: velvet_result.final_loss,
        adamw_final_loss: adamw_result.final_loss,
        velvet_final_ppl: velvet_result.perplexity_history.last().copied().unwrap_or(0.0),
        adamw_final_ppl: adamw_result.perplexity_history.last().copied().unwrap_or(0.0),
    };

    // Print results
    println!("\n========================================");
    println!("         BENCHMARK RESULTS");
    println!("========================================\n");
    println!("  {:>20} {:>12} {:>12}", "", "AdamW", "Velvet");
    println!("  {:>20} {:>12} {:>12}", "---", "---", "---");
    println!("  {:>20} {:>12.4} {:>12.4}", "Final Loss",
        adamw_result.final_loss, velvet_result.final_loss);
    println!("  {:>20} {:>12.4} {:>12.4}", "Best Loss",
        adamw_result.best_loss, velvet_result.best_loss);
    println!("  {:>20} {:>12.2} {:>12.2}", "Final Perplexity",
        comparison.adamw_final_ppl, comparison.velvet_final_ppl);
    println!("  {:>20} {:>12.1}s {:>11.1}s", "Training Time",
        adamw_result.training_time_secs, velvet_result.training_time_secs);
    println!();
    println!("  Loss improvement:       {:.1}%", comparison.loss_improvement_pct);
    println!("  Perplexity improvement: {:.1}%", comparison.perplexity_improvement_pct);
    println!();

    // Save report
    let report = BenchmarkReport {
        velvet: velvet_result,
        adamw: adamw_result,
        comparison,
    };
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&output, &json)?;
    println!("  Report saved to: {}", output.display());

    Ok(())
}

fn run_adamw(
    config: &VesperConfig,
    dataset: &vesper_training::Dataset,
    device: &Device,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    _seq_len: usize,
) -> Result<BenchmarkResult> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = VesperLM::new(config.clone(), vb)?;

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        },
    )?;

    let num_batches = (dataset.len() + batch_size - 1) / batch_size;
    let mut loss_history = Vec::new();
    let mut perplexity_history = Vec::new();
    let mut step_metrics = Vec::new();
    let mut best_loss = f32::MAX;
    let mut global_step: usize = 0;

    let start = Instant::now();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut count = 0;

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("    [{bar:40.cyan/blue}] {pos}/{len} [{elapsed}<{eta}] {msg}")
            .unwrap()
            .progress_chars("=>-"));

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(dataset.len());

            let (input_ids, attention_mask, labels) = prepare_batch(dataset, start_idx, end_idx, device)?;
            let logits = model.forward(&input_ids, Some(&attention_mask))?;
            let loss = compute_loss(&logits, &labels)?;
            let loss_val = loss.to_scalar::<f32>()?;

            optimizer.backward_step(&loss)?;

            step_metrics.push(StepMetric {
                step: global_step,
                loss: loss_val,
                entropy_scale: None,
                ppl_scale: None,
                grad_norm: None,
            });
            global_step += 1;

            epoch_loss += loss_val;
            count += 1;

            if batch_idx % 50 == 0 {
                let avg_so_far = epoch_loss / count as f32;
                pb.set_message(format!("loss: {:.4}", avg_so_far));
            }
            pb.inc(1);
        }

        let avg = epoch_loss / count.max(1) as f32;
        let ppl = (avg as f64).exp() as f32;
        loss_history.push(avg);
        perplexity_history.push(ppl);
        if avg < best_loss { best_loss = avg; }

        pb.finish_and_clear();
        println!("    Epoch {}/{}: loss={:.4} ppl={:.2}", epoch + 1, epochs, avg, ppl);
    }

    Ok(BenchmarkResult {
        optimizer: "AdamW".to_string(),
        model_size: format!("{}params", config.total_params()),
        epochs,
        final_loss: loss_history.last().copied().unwrap_or(0.0),
        best_loss,
        training_time_secs: start.elapsed().as_secs_f64(),
        loss_history,
        perplexity_history,
        step_metrics,
    })
}

fn run_velvet(
    config: &VesperConfig,
    dataset: &vesper_training::Dataset,
    device: &Device,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    _seq_len: usize,
) -> Result<BenchmarkResult> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = VesperLM::new(config.clone(), vb)?;

    let velvet_config = VelvetConfig {
        lr,
        ..VelvetConfig::optimal()
    };
    let mut optimizer = VelvetOptimizer::new(velvet_config);

    let num_batches = (dataset.len() + batch_size - 1) / batch_size;
    let mut loss_history = Vec::new();
    let mut perplexity_history = Vec::new();
    let mut step_metrics = Vec::new();
    let mut best_loss = f32::MAX;
    let mut global_step: usize = 0;
    let vocab_size = config.vocab_size;

    let start = Instant::now();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut count = 0;

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("    [{bar:40.green/black}] {pos}/{len} [{elapsed}<{eta}] {msg}")
            .unwrap()
            .progress_chars("=>-"));

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(dataset.len());

            let (input_ids, attention_mask, labels) = prepare_batch(dataset, start_idx, end_idx, device)?;
            let logits = model.forward(&input_ids, Some(&attention_mask))?;
            let loss = compute_loss(&logits, &labels)?;
            let loss_val = loss.to_scalar::<f32>()?;

            // 1. Entropy-Adaptive LR: compute REAL logits entropy
            let current_entropy = compute_logits_entropy(&logits, vocab_size)?;
            let baseline_entropy = 0.5;
            let entropy_scale = (current_entropy / baseline_entropy).clamp(0.5, 2.0);
            optimizer.set_entropy_scale(entropy_scale);

            // 2. Perplexity-Guided Momentum: baseline_ppl=40.0
            let current_ppl = (loss_val as f64).exp();
            let baseline_ppl = 40.0;
            let ppl_scale = (baseline_ppl / current_ppl.max(1.0)).clamp(0.5, 2.0);
            optimizer.set_perplexity_scale(ppl_scale);

            // Backward + optimizer step (single call, like AdamW's backward_step)
            optimizer.backward_step(&loss, &varmap)?;

            step_metrics.push(StepMetric {
                step: global_step,
                loss: loss_val,
                entropy_scale: Some(entropy_scale as f32),
                ppl_scale: Some(ppl_scale as f32),
                grad_norm: Some(optimizer.last_grad_norm() as f32),
            });
            global_step += 1;

            epoch_loss += loss_val;
            count += 1;

            if batch_idx % 50 == 0 {
                let avg_so_far = epoch_loss / count as f32;
                pb.set_message(format!("loss: {:.4}", avg_so_far));
            }
            pb.inc(1);
        }

        let avg = epoch_loss / count.max(1) as f32;
        let ppl = (avg as f64).exp() as f32;
        loss_history.push(avg);
        perplexity_history.push(ppl);
        if avg < best_loss { best_loss = avg; }

        pb.finish_and_clear();
        println!("    Epoch {}/{}: loss={:.4} ppl={:.2}", epoch + 1, epochs, avg, ppl);
    }

    Ok(BenchmarkResult {
        optimizer: "Velvet".to_string(),
        model_size: format!("{}params", config.total_params()),
        epochs,
        final_loss: loss_history.last().copied().unwrap_or(0.0),
        best_loss,
        training_time_secs: start.elapsed().as_secs_f64(),
        loss_history,
        perplexity_history,
        step_metrics,
    })
}

fn prepare_batch(
    dataset: &vesper_training::Dataset,
    start_idx: usize,
    end_idx: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
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

    let input_ids = Tensor::new(input_ids_batch, device)?;
    let attention_mask = Tensor::new(attention_mask_batch, device)?;
    let labels = Tensor::new(labels_batch, device)?;

    Ok((input_ids, attention_mask, labels))
}

fn compute_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    let logits_2d = logits.reshape((batch_size * seq_len, vocab_size))?;
    let labels_1d = labels.flatten_all()?;
    cross_entropy(&logits_2d, &labels_1d)
        .map_err(|e| anyhow::anyhow!("Cross entropy failed: {}", e))
}

/// Compute normalized entropy from logits (matches Python reference).
/// Returns mean(entropy / log(vocab_size)), in [0, 1] range.
/// Uses detached logits to avoid polluting the backward computation graph.
fn compute_logits_entropy(logits: &Tensor, vocab_size: usize) -> Result<f64> {
    let max_entropy = (vocab_size as f64).ln();
    let last_dim = logits.dims().len() - 1;

    // Detach to avoid tracking in computation graph
    let logits_d = logits.detach();

    // Softmax probabilities
    let probs = candle_nn::ops::softmax(&logits_d, last_dim)?;

    // log(probs) with numerical safety (clamp to avoid log(0))
    let log_probs = probs.clamp(1e-10, 1.0)?.log()?;

    // entropy = -sum(p * log(p), dim=-1) -> [batch, seq]
    let entropy = (probs.mul(&log_probs)?.sum(last_dim)? * -1.0)?;

    // Normalize by max entropy -> [0, 1]
    let normalized = (entropy / max_entropy)?;

    // Mean over all positions
    let mean = normalized.mean_all()?.to_scalar::<f32>()? as f64;

    Ok(mean)
}
