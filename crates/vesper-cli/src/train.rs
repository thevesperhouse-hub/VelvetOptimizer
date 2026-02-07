//! Train subcommand - Full training pipeline

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{self, loss::cross_entropy, VarMap, VarBuilder};
use std::path::PathBuf;
use std::time::Instant;

use vesper_core::{VesperConfig, VesperLM, MappedDataset};
use vesper_optimizer::{VelvetOptimizer, VelvetConfig};
use vesper_training::{Trainer, TrainerConfig, DatasetLoader, StreamingTextLoader};

use crate::tokenizer;

/// Run the train subcommand
pub fn run(
    dataset: PathBuf,
    format: String,
    tokenizer_name: String,
    model_size: String,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    seq_len: usize,
    save_every: usize,
    output_dir: PathBuf,
    max_steps: usize,
    vocab_size_override: usize,
    cache_path: Option<PathBuf>,
    streaming: bool,
    chunk_mb: usize,
) -> Result<()> {
    println!("\n=== VesperAI Training ===\n");

    // 1. Select device
    let device = if candle_core::utils::cuda_is_available() {
        println!("  CUDA available, using GPU");
        Device::new_cuda(0)?
    } else {
        println!("  No CUDA, using CPU");
        Device::Cpu
    };

    // 2. Load tokenizer
    let tok = tokenizer::load_tokenizer(&tokenizer_name)?;
    let vocab_size = if vocab_size_override > 0 {
        vocab_size_override
    } else {
        tokenizer::vocab_size(&tok)
    };

    // 3. Build model config
    let config = match model_size.as_str() {
        "tiny" => VesperConfig::tiny(),
        "small" => VesperConfig::small(),
        "medium" => VesperConfig::medium(),
        "large" => VesperConfig::large(),
        "xlarge" | "1b" => VesperConfig::xlarge(),
        _ => anyhow::bail!("Unknown model size: {}. Use: tiny, small, medium, large, xlarge", model_size),
    };
    let config = config.with_vocab_size(vocab_size);
    config.validate()?;

    println!("\n  Model: {} ({} params)", model_size, format_params(config.total_params()));
    println!("  Vocab size: {}", vocab_size);

    // 4. Create model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = VesperLM::new(config.clone(), vb)?;
    println!("  Model initialized on {:?}", device);

    // 5. Create optimizer
    let velvet_config = VelvetConfig {
        lr,
        ..VelvetConfig::optimal()
    };
    let mut optimizer = VelvetOptimizer::new(velvet_config);

    // 6. Route to appropriate training mode
    if let Some(ref cache) = cache_path {
        // === CACHE MODE: pre-built binary cache ===
        println!("\n  Mode: Binary cache");
        println!("  Loading cache: {}", cache.display());
        let mapped = MappedDataset::load(cache)
            .context("Failed to load binary cache")?;
        println!("  Cache: {} sequences, {} tokens",
            mapped.num_sequences(), mapped.total_tokens());

        run_cached_training(
            &model, &varmap, &mut optimizer, &mapped, &device,
            epochs, batch_size, seq_len, save_every, &output_dir, max_steps,
        )?;
    } else if streaming {
        // === STREAMING MODE: read large text files in chunks ===
        println!("\n  Mode: Streaming ({}MB chunks)", chunk_mb);
        let mut loader = StreamingTextLoader::new(&dataset, tok, seq_len, chunk_mb)?;

        run_streaming_training(
            &model, &varmap, &mut optimizer, &mut loader, &device,
            epochs, batch_size, seq_len, save_every, &output_dir, max_steps,
        )?;
    } else {
        // === DEFAULT MODE: load entire dataset into memory ===
        println!("\n  Mode: In-memory");
        let loader = DatasetLoader::from_tokenizer(tok, seq_len);

        let detected_format = if format == "auto" {
            detect_format(&dataset)
        } else {
            format.clone()
        };

        println!("  Loading dataset: {} (format: {})", dataset.display(), detected_format);
        let start_load = Instant::now();

        let mut ds = match detected_format.as_str() {
            "text" | "txt" => loader.load_text(&dataset)
                .context("Failed to load text dataset")?,
            "jsonl" => loader.load_jsonl(&dataset)
                .context("Failed to load JSONL dataset")?,
            "squad" | "json" => loader.load_squad(&dataset)
                .context("Failed to load SQuAD dataset")?,
            _ => anyhow::bail!("Unknown dataset format: {}. Use: text, jsonl, squad, auto", detected_format),
        };

        println!("  Dataset loaded in {:.1}s ({} samples)",
            start_load.elapsed().as_secs_f64(), ds.len());

        // Use the existing Trainer for in-memory mode
        let trainer_config = TrainerConfig {
            num_epochs: epochs,
            batch_size,
            learning_rate: lr,
            log_interval: 10,
            save_interval: save_every,
            output_dir: output_dir.to_string_lossy().to_string(),
            max_steps: if max_steps > 0 { Some(max_steps) } else { None },
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
        };

        let trainer = Trainer::new(trainer_config, device);
        let start_train = Instant::now();
        let metrics = trainer.train(&model, &varmap, &mut optimizer, &mut ds)?;
        let train_time = start_train.elapsed();

        println!("\n=== Training Complete ===");
        println!("  Total time: {:.1}s", train_time.as_secs_f64());
        println!("  Epochs: {}", metrics.epochs.len());
        if let Some(last) = metrics.epochs.last() {
            println!("  Final loss: {:.4}", last.avg_loss);
        }
        println!("  Checkpoints saved to: {}", output_dir.display());
    }

    Ok(())
}

/// Train from a pre-built binary cache (MappedDataset)
fn run_cached_training(
    model: &VesperLM,
    varmap: &VarMap,
    optimizer: &mut VelvetOptimizer,
    mapped: &MappedDataset,
    device: &Device,
    epochs: usize,
    batch_size: usize,
    seq_len: usize,
    save_every: usize,
    output_dir: &PathBuf,
    max_steps: usize,
) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};

    std::fs::create_dir_all(output_dir)?;
    let vocab_size = model.config().vocab_size;
    let num_sequences = mapped.num_sequences();
    let num_batches = (num_sequences + batch_size - 1) / batch_size;
    let mut global_step: usize = 0;

    let start = Instant::now();

    for epoch in 0..epochs {
        println!("\n[Epoch {}/{}]", epoch + 1, epochs);

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("  {bar:40.green/black} {pos}/{len} [{elapsed}<{eta}] {msg}")
            .unwrap());

        let mut epoch_loss = 0.0;
        let mut count = 0;

        for batch_idx in 0..num_batches {
            if max_steps > 0 && global_step >= max_steps {
                break;
            }

            // Collect batch indices
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(num_sequences);
            let indices: Vec<usize> = (start_idx..end_idx).collect();

            let batch_seqs = mapped.get_batch(&indices, seq_len);
            if batch_seqs.is_empty() { continue; }

            // Build tensors
            let input_ids_data: Vec<Vec<u32>> = batch_seqs.iter()
                .map(|s| s[..s.len().min(seq_len)].to_vec())
                .collect();
            let labels_data: Vec<Vec<i64>> = batch_seqs.iter()
                .map(|s| s[1..s.len().min(seq_len + 1)].iter().map(|&id| id as i64).collect())
                .collect();

            let actual_len = input_ids_data[0].len();
            let input_ids = Tensor::new(input_ids_data, device)?;
            let attention_mask = Tensor::ones((indices.len(), actual_len), DType::U32, device)?;
            let labels = Tensor::new(labels_data, device)?;

            // Forward
            let logits = model.forward(&input_ids, Some(&attention_mask))?;
            let loss = compute_loss(&logits, &labels)?;
            let loss_val = loss.to_scalar::<f32>()?;

            // Adaptive scales
            let current_entropy = compute_logits_entropy(&logits, vocab_size)?;
            let entropy_scale = (current_entropy / 0.5).clamp(0.5, 2.0);
            optimizer.set_entropy_scale(entropy_scale);

            let current_ppl = (loss_val as f64).exp();
            let ppl_scale = (40.0 / current_ppl.max(1.0)).clamp(0.5, 2.0);
            optimizer.set_perplexity_scale(ppl_scale);

            optimizer.backward_step(&loss, varmap)?;

            epoch_loss += loss_val;
            count += 1;
            global_step += 1;

            if batch_idx % 50 == 0 {
                pb.set_message(format!("loss: {:.4}", epoch_loss / count as f32));
            }
            pb.inc(1);

            if save_every > 0 && global_step % save_every == 0 {
                let path = format!("{}/checkpoint-{}.safetensors",
                    output_dir.display(), global_step);
                varmap.save(&path)?;
                pb.println(format!("  Checkpoint saved: {}", path));
            }
        }

        let avg = epoch_loss / count.max(1) as f32;
        let ppl = (avg as f64).exp() as f32;
        pb.finish_and_clear();
        println!("  Epoch {}/{}: loss={:.4} ppl={:.2}", epoch + 1, epochs, avg, ppl);

        if max_steps > 0 && global_step >= max_steps {
            println!("  Reached max_steps ({}), stopping.", max_steps);
            break;
        }
    }

    // Final checkpoint
    let final_path = format!("{}/checkpoint-final.safetensors", output_dir.display());
    varmap.save(&final_path)?;
    println!("\n  Final checkpoint: {}", final_path);
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

/// Train in streaming mode (chunk-by-chunk reading of large files)
fn run_streaming_training(
    model: &VesperLM,
    varmap: &VarMap,
    optimizer: &mut VelvetOptimizer,
    loader: &mut StreamingTextLoader,
    device: &Device,
    epochs: usize,
    batch_size: usize,
    _seq_len: usize,
    save_every: usize,
    output_dir: &PathBuf,
    max_steps: usize,
) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};

    std::fs::create_dir_all(output_dir)?;
    let vocab_size = model.config().vocab_size;
    let mut global_step: usize = 0;

    let start = Instant::now();

    for epoch in 0..epochs {
        println!("\n[Epoch {}/{}] (streaming)", epoch + 1, epochs);
        loader.reset()?;

        let mut epoch_loss = 0.0;
        let mut epoch_count = 0;
        let mut chunk_idx = 0;

        while let Some(chunk_dataset) = loader.next_chunk()? {
            chunk_idx += 1;
            let num_batches = (chunk_dataset.len() + batch_size - 1) / batch_size;

            let pb = ProgressBar::new(num_batches as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template(&format!("  chunk {} [{{bar:30.green/black}}] {{pos}}/{{len}} [{{elapsed}}] {{msg}}", chunk_idx))
                .unwrap());

            for batch_idx in 0..num_batches {
                if max_steps > 0 && global_step >= max_steps {
                    break;
                }

                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(chunk_dataset.len());

                let mut input_ids_batch = Vec::new();
                let mut attention_mask_batch = Vec::new();
                let mut labels_batch = Vec::new();

                for idx in start_idx..end_idx {
                    if let Some(sample) = chunk_dataset.get(idx) {
                        input_ids_batch.push(sample.input_ids.clone());
                        attention_mask_batch.push(sample.attention_mask.clone());
                        labels_batch.push(sample.labels.clone());
                    }
                }

                if input_ids_batch.is_empty() { continue; }

                let input_ids = Tensor::new(input_ids_batch, device)?;
                let attention_mask = Tensor::new(attention_mask_batch, device)?;
                let labels = Tensor::new(labels_batch, device)?;

                let logits = model.forward(&input_ids, Some(&attention_mask))?;
                let loss = compute_loss(&logits, &labels)?;
                let loss_val = loss.to_scalar::<f32>()?;

                // Adaptive scales
                let current_entropy = compute_logits_entropy(&logits, vocab_size)?;
                let entropy_scale = (current_entropy / 0.5).clamp(0.5, 2.0);
                optimizer.set_entropy_scale(entropy_scale);

                let current_ppl = (loss_val as f64).exp();
                let ppl_scale = (40.0 / current_ppl.max(1.0)).clamp(0.5, 2.0);
                optimizer.set_perplexity_scale(ppl_scale);

                optimizer.backward_step(&loss, varmap)?;

                epoch_loss += loss_val;
                epoch_count += 1;
                global_step += 1;

                if batch_idx % 50 == 0 {
                    pb.set_message(format!("loss: {:.4}", epoch_loss / epoch_count as f32));
                }
                pb.inc(1);

                if save_every > 0 && global_step % save_every == 0 {
                    let path = format!("{}/checkpoint-{}.safetensors",
                        output_dir.display(), global_step);
                    varmap.save(&path)?;
                    pb.println(format!("  Checkpoint saved: {}", path));
                }
            }

            pb.finish_and_clear();

            if max_steps > 0 && global_step >= max_steps {
                break;
            }
        }

        let avg = epoch_loss / epoch_count.max(1) as f32;
        let ppl = (avg as f64).exp() as f32;
        println!("  Epoch {}/{}: loss={:.4} ppl={:.2} ({} chunks, {} steps)",
            epoch + 1, epochs, avg, ppl, chunk_idx, epoch_count);

        if max_steps > 0 && global_step >= max_steps {
            println!("  Reached max_steps ({}), stopping.", max_steps);
            break;
        }
    }

    // Final checkpoint
    let final_path = format!("{}/checkpoint-final.safetensors", output_dir.display());
    varmap.save(&final_path)?;
    println!("\n  Final checkpoint: {}", final_path);
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

fn compute_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    let logits_2d = logits.reshape((batch_size * seq_len, vocab_size))?;
    let labels_1d = labels.flatten_all()?;
    cross_entropy(&logits_2d, &labels_1d)
        .map_err(|e| anyhow::anyhow!("Cross entropy failed: {}", e))
}

fn compute_logits_entropy(logits: &Tensor, vocab_size: usize) -> Result<f64> {
    let max_entropy = (vocab_size as f64).ln();
    let last_dim = logits.dims().len() - 1;
    let logits_d = logits.detach();
    let probs = candle_nn::ops::softmax(&logits_d, last_dim)?;
    let log_probs = probs.clamp(1e-10, 1.0)?.log()?;
    let entropy = (probs.mul(&log_probs)?.sum(last_dim)? * -1.0)?;
    let normalized = (entropy / max_entropy)?;
    let mean = normalized.mean_all()?.to_scalar::<f32>()? as f64;
    Ok(mean)
}

fn detect_format(path: &PathBuf) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => "jsonl".to_string(),
        Some("json") => "squad".to_string(),
        Some("txt") | Some("text") => "text".to_string(),
        _ => "text".to_string(),
    }
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
