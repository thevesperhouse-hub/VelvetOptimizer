//! Train subcommand - Full training pipeline
//!
//! Supports:
//! - In-memory, streaming, and binary cache dataset modes
//! - Velvet or AdamW optimizer (--optimizer)
//! - Resume from checkpoint (--resume)
//! - Auto-save on SIGTERM/SIGINT (Vast.ai preemption safe)
//! - Periodic checkpointing (--save-every)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{self, loss::cross_entropy, VarMap, VarBuilder, Optimizer, optim::AdamW, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use vesper_core::{VesperConfig, VesperLM, MappedDataset};
use vesper_optimizer::{VelvetOptimizer, VelvetConfig};
use vesper_training::{DatasetLoader, StreamingTextLoader};

use crate::tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingLog {
    optimizer: String,
    model_size: String,
    steps: Vec<StepLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepLog {
    step: usize,
    loss: f32,
}

/// Which optimizer to use
enum OptimizerKind {
    Velvet(VelvetOptimizer),
    AdamW(AdamW),
}

impl OptimizerKind {
    fn step(
        &mut self,
        loss: &Tensor,
        varmap: &VarMap,
        logits: &Tensor,
        loss_val: f32,
        vocab_size: usize,
    ) -> Result<()> {
        match self {
            OptimizerKind::Velvet(opt) => {
                // Entropy-Adaptive LR
                let current_entropy = compute_logits_entropy(logits, vocab_size)?;
                let entropy_scale = (current_entropy / 0.5).clamp(0.5, 2.0);
                opt.set_entropy_scale(entropy_scale);

                // Perplexity-Guided Momentum
                let current_ppl = (loss_val as f64).exp();
                let ppl_scale = (40.0 / current_ppl.max(1.0)).clamp(0.5, 2.0);
                opt.set_perplexity_scale(ppl_scale);

                opt.backward_step(loss, varmap)?;
            }
            OptimizerKind::AdamW(opt) => {
                opt.backward_step(loss)?;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        match self {
            OptimizerKind::Velvet(_) => "Velvet",
            OptimizerKind::AdamW(_) => "AdamW",
        }
    }
}

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
    optimizer_name: String,
    resume: Option<PathBuf>,
    moe: bool,
    num_experts: usize,
    top_k: usize,
) -> Result<()> {
    println!("\n=== VesperAI Training ===\n");

    // Setup SIGTERM/SIGINT handler for graceful shutdown
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        eprintln!("\n  SIGTERM/SIGINT received! Saving checkpoint before exit...");
        shutdown_clone.store(true, Ordering::SeqCst);
    }).ok();

    // 1. Select device
    let device = if candle_core::utils::cuda_is_available() {
        println!("  CUDA available, using GPU");
        Device::new_cuda(0)?
    } else {
        println!("  No CUDA, using CPU");
        Device::Cpu
    };

    // 2. Build model config (needed to resolve tokenizer)
    let config = match model_size.as_str() {
        "tiny" => VesperConfig::tiny(),
        "small" => VesperConfig::small(),
        "medium" => VesperConfig::medium(),
        "tiny-moe" => VesperConfig::tiny_moe(),
        "medium-moe" => VesperConfig::medium_moe(),
        "large" => VesperConfig::large(),
        "large-moe" => VesperConfig::large_moe(),
        "xlarge" | "1b" => VesperConfig::xlarge(),
        _ => anyhow::bail!("Unknown model size: {}. Use: tiny, small, medium, medium-moe, large, large-moe, xlarge", model_size),
    };
    // Apply --moe flag (overrides preset if both specified)
    let config = if moe { config.with_moe(num_experts, top_k) } else { config };
    // 3. Resolve tokenizer ("auto" picks based on model size)
    let resolved_tokenizer = resolve_tokenizer(&tokenizer_name, &model_size);
    let tok = tokenizer::load_tokenizer(&resolved_tokenizer)?;
    let vocab_size = if vocab_size_override > 0 {
        vocab_size_override
    } else {
        tokenizer::vocab_size(&tok)
    };

    let config = config.with_vocab_size(vocab_size);
    config.validate()?;

    let moe_str = if config.moe_enabled {
        format!(" [MoE: {}x top-{} experts]", config.moe_num_experts, config.moe_top_k)
    } else {
        String::new()
    };
    println!("  Model: {} ({} params){}", model_size, format_params(config.total_params()), moe_str);
    println!("  Vocab size: {}", vocab_size);

    // 4. Create model + optionally resume from checkpoint
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = VesperLM::new(config.clone(), vb)?;

    let start_step = if let Some(ref ckpt) = resume {
        println!("  Resuming from checkpoint: {}", ckpt.display());
        varmap.load(ckpt)?;
        let step = detect_step_from_path(ckpt);
        println!("  Weights loaded successfully (resuming from step {})", step);
        step
    } else {
        println!("  Model initialized from scratch");
        0
    };
    println!("  Device: {:?}", device);

    // 5. Create optimizer
    let mut optimizer = match optimizer_name.to_lowercase().as_str() {
        "velvet" => {
            let velvet_config = VelvetConfig {
                lr,
                ..VelvetConfig::optimal()
            };
            OptimizerKind::Velvet(VelvetOptimizer::new(velvet_config))
        }
        "adamw" => {
            OptimizerKind::AdamW(AdamW::new(
                varmap.all_vars(),
                ParamsAdamW {
                    lr,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.01,
                },
            )?)
        }
        _ => anyhow::bail!("Unknown optimizer: {}. Use: velvet, adamw", optimizer_name),
    };
    println!("  Optimizer: {}", optimizer.name());

    // 6. Route to appropriate training mode
    std::fs::create_dir_all(&output_dir)?;

    if let Some(ref cache) = cache_path {
        println!("\n  Mode: Binary cache");
        println!("  Loading cache: {}", cache.display());
        let mapped = MappedDataset::load(cache)
            .context("Failed to load binary cache")?;
        println!("  Cache: {} sequences, {} tokens",
            mapped.num_sequences(), mapped.total_tokens());

        run_training_loop(
            &model, &varmap, &mut optimizer, &device,
            epochs, batch_size, seq_len, save_every, &output_dir, max_steps,
            &shutdown, start_step, &optimizer_name, &model_size,
            TrainingData::Cached(&mapped),
        )?;
    } else if streaming {
        println!("\n  Mode: Streaming ({}MB chunks)", chunk_mb);
        let mut loader = StreamingTextLoader::new(&dataset, tok, seq_len, chunk_mb)?;

        run_streaming_loop(
            &model, &varmap, &mut optimizer, &mut loader, &device,
            epochs, batch_size, save_every, &output_dir, max_steps,
            &shutdown, start_step, &optimizer_name, &model_size,
        )?;
    } else {
        println!("\n  Mode: In-memory");
        let loader = DatasetLoader::from_tokenizer(tok, seq_len);

        let detected_format = if format == "auto" {
            detect_format(&dataset)
        } else {
            format.clone()
        };

        println!("  Loading dataset: {} (format: {})", dataset.display(), detected_format);
        let start_load = Instant::now();

        let ds = match detected_format.as_str() {
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

        run_training_loop(
            &model, &varmap, &mut optimizer, &device,
            epochs, batch_size, seq_len, save_every, &output_dir, max_steps,
            &shutdown, start_step, &optimizer_name, &model_size,
            TrainingData::InMemory(&ds),
        )?;
    }

    Ok(())
}

enum TrainingData<'a> {
    InMemory(&'a vesper_training::Dataset),
    Cached(&'a MappedDataset),
}

/// Unified training loop for in-memory and cached modes
fn run_training_loop(
    model: &VesperLM,
    varmap: &VarMap,
    optimizer: &mut OptimizerKind,
    device: &Device,
    epochs: usize,
    batch_size: usize,
    seq_len: usize,
    save_every: usize,
    output_dir: &PathBuf,
    max_steps: usize,
    shutdown: &Arc<AtomicBool>,
    start_step: usize,
    optimizer_name: &str,
    model_size: &str,
    data: TrainingData,
) -> Result<()> {
    let vocab_size = model.config().vocab_size;
    let num_samples = match &data {
        TrainingData::InMemory(ds) => ds.len(),
        TrainingData::Cached(m) => m.num_sequences(),
    };
    let num_batches = (num_samples + batch_size - 1) / batch_size;
    let mut global_step: usize = start_step;
    let mut log = TrainingLog {
        optimizer: optimizer_name.to_string(),
        model_size: model_size.to_string(),
        steps: Vec::new(),
    };

    // Load existing log if resuming
    let log_path = format!("{}/training_log.json", output_dir.display());
    if start_step > 0 {
        if let Ok(data) = std::fs::read_to_string(&log_path) {
            if let Ok(existing) = serde_json::from_str::<TrainingLog>(&data) {
                log.steps = existing.steps;
                println!("  Resumed training log ({} existing entries)", log.steps.len());
            }
        }
    }

    let start = Instant::now();

    println!("  Samples: {}, Batches/epoch: {}, Start step: {}\n", num_samples, num_batches, start_step);

    for epoch in 0..epochs {
        println!("[Epoch {}/{}]", epoch + 1, epochs);

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("  {bar:40.green/black} {pos}/{len} [{elapsed}<{eta}] {msg}")
            .unwrap());

        let mut epoch_loss = 0.0;
        let mut count = 0;

        for batch_idx in 0..num_batches {
            // Check shutdown signal (Vast.ai preemption)
            if shutdown.load(Ordering::SeqCst) {
                let path = format!("{}/checkpoint-emergency-step{}.safetensors",
                    output_dir.display(), global_step);
                varmap.save(&path)?;
                save_log(&log, &log_path)?;
                println!("\n  Emergency checkpoint saved: {}", path);
                println!("  Training interrupted at step {} (epoch {}/{})",
                    global_step, epoch + 1, epochs);
                return Ok(());
            }

            if max_steps > 0 && global_step >= max_steps {
                break;
            }

            // Build batch tensors
            let (input_ids, attention_mask, labels) = match &data {
                TrainingData::InMemory(ds) => {
                    let s = batch_idx * batch_size;
                    let e = (s + batch_size).min(ds.len());
                    prepare_batch_inmemory(ds, s, e, device)?
                }
                TrainingData::Cached(m) => {
                    let s = batch_idx * batch_size;
                    let e = (s + batch_size).min(m.num_sequences());
                    prepare_batch_cached(m, s, e, seq_len, device)?
                }
            };

            // Forward
            let (logits, aux_loss_opt) = model.forward(&input_ids, Some(&attention_mask))?;
            let mut loss = compute_loss(&logits, &labels)?;

            // Add MoE auxiliary loss if present
            if let Some(aux_loss) = aux_loss_opt {
                loss = loss.add(&(aux_loss * model.config().moe_aux_loss_weight)?)?;
            }
            let loss_val = loss.to_scalar::<f32>()?;

            // Backward + optimizer step
            optimizer.step(&loss, varmap, &logits, loss_val, vocab_size)?;

            epoch_loss += loss_val;
            count += 1;
            global_step += 1;

            log.steps.push(StepLog { step: global_step, loss: loss_val });

            if batch_idx % 50 == 0 {
                pb.set_message(format!("loss: {:.4} | step: {}", epoch_loss / count as f32, global_step));
            }
            pb.inc(1);

            // Periodic checkpoint + save log
            if save_every > 0 && global_step % save_every == 0 {
                let path = format!("{}/checkpoint-{}.safetensors",
                    output_dir.display(), global_step);
                varmap.save(&path)?;
                save_log(&log, &log_path)?;
                pb.println(format!("  Checkpoint saved: {} (step {})", path, global_step));
            }
        }

        let avg = epoch_loss / count.max(1) as f32;
        let ppl = (avg as f64).exp() as f32;
        pb.finish_and_clear();
        println!("  Epoch {}/{}: loss={:.4} ppl={:.2} ({} steps)\n",
            epoch + 1, epochs, avg, ppl, count);

        // Epoch-end checkpoint + save log
        let path = format!("{}/checkpoint-epoch{}.safetensors",
            output_dir.display(), epoch + 1);
        varmap.save(&path)?;
        save_log(&log, &log_path)?;
        println!("  Epoch checkpoint: {}", path);

        if max_steps > 0 && global_step >= max_steps {
            println!("  Reached max_steps ({}), stopping.", max_steps);
            break;
        }
    }

    // Final checkpoint + save log
    let final_path = format!("{}/checkpoint-final.safetensors", output_dir.display());
    varmap.save(&final_path)?;
    save_log(&log, &log_path)?;
    println!("\n=== Training Complete ===");
    println!("  Final checkpoint: {}", final_path);
    println!("  Training log: {}", log_path);
    println!("  Total steps: {}", global_step);
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

/// Streaming training loop (chunk-by-chunk)
fn run_streaming_loop(
    model: &VesperLM,
    varmap: &VarMap,
    optimizer: &mut OptimizerKind,
    loader: &mut StreamingTextLoader,
    device: &Device,
    epochs: usize,
    batch_size: usize,
    save_every: usize,
    output_dir: &PathBuf,
    max_steps: usize,
    shutdown: &Arc<AtomicBool>,
    start_step: usize,
    optimizer_name: &str,
    model_size: &str,
) -> Result<()> {
    let vocab_size = model.config().vocab_size;
    let mut global_step: usize = start_step;
    let mut log = TrainingLog {
        optimizer: optimizer_name.to_string(),
        model_size: model_size.to_string(),
        steps: Vec::new(),
    };
    let log_path = format!("{}/training_log.json", output_dir.display());

    if start_step > 0 {
        if let Ok(data) = std::fs::read_to_string(&log_path) {
            if let Ok(existing) = serde_json::from_str::<TrainingLog>(&data) {
                log.steps = existing.steps;
            }
        }
    }

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
                // Check shutdown
                if shutdown.load(Ordering::SeqCst) {
                    let path = format!("{}/checkpoint-emergency-step{}.safetensors",
                        output_dir.display(), global_step);
                    varmap.save(&path)?;
                    save_log(&log, &log_path)?;
                    println!("\n  Emergency checkpoint saved: {}", path);
                    return Ok(());
                }

                if max_steps > 0 && global_step >= max_steps {
                    break;
                }

                let s = batch_idx * batch_size;
                let e = (s + batch_size).min(chunk_dataset.len());
                let (input_ids, attention_mask, labels) =
                    prepare_batch_inmemory(&chunk_dataset, s, e, device)?;

                let (logits, aux_loss_opt) = model.forward(&input_ids, Some(&attention_mask))?;
                let mut loss = compute_loss(&logits, &labels)?;
                if let Some(aux_loss) = aux_loss_opt {
                    loss = loss.add(&(aux_loss * model.config().moe_aux_loss_weight)?)?;
                }
                let loss_val = loss.to_scalar::<f32>()?;

                optimizer.step(&loss, varmap, &logits, loss_val, vocab_size)?;

                epoch_loss += loss_val;
                epoch_count += 1;
                global_step += 1;

                log.steps.push(StepLog { step: global_step, loss: loss_val });

                if batch_idx % 50 == 0 {
                    pb.set_message(format!("loss: {:.4}", epoch_loss / epoch_count as f32));
                }
                pb.inc(1);

                if save_every > 0 && global_step % save_every == 0 {
                    let path = format!("{}/checkpoint-{}.safetensors",
                        output_dir.display(), global_step);
                    varmap.save(&path)?;
                    save_log(&log, &log_path)?;
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

        // Epoch checkpoint + log
        let path = format!("{}/checkpoint-epoch{}.safetensors",
            output_dir.display(), epoch + 1);
        varmap.save(&path)?;
        save_log(&log, &log_path)?;
        println!("  Epoch checkpoint: {}", path);

        if max_steps > 0 && global_step >= max_steps {
            println!("  Reached max_steps ({}), stopping.", max_steps);
            break;
        }
    }

    let final_path = format!("{}/checkpoint-final.safetensors", output_dir.display());
    varmap.save(&final_path)?;
    save_log(&log, &log_path)?;
    println!("\n=== Training Complete ===");
    println!("  Final checkpoint: {}", final_path);
    println!("  Training log: {}", log_path);
    println!("  Total steps: {}", global_step);
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

fn prepare_batch_inmemory(
    ds: &vesper_training::Dataset,
    start: usize,
    end: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut input_ids = Vec::new();
    let mut masks = Vec::new();
    let mut labels = Vec::new();

    for idx in start..end {
        if let Some(s) = ds.get(idx) {
            input_ids.push(s.input_ids.clone());
            masks.push(s.attention_mask.clone());
            labels.push(s.labels.clone());
        }
    }

    Ok((
        Tensor::new(input_ids, device)?,
        Tensor::new(masks, device)?,
        Tensor::new(labels, device)?,
    ))
}

fn prepare_batch_cached(
    mapped: &MappedDataset,
    start: usize,
    end: usize,
    seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let indices: Vec<usize> = (start..end).collect();
    let batch_seqs = mapped.get_batch(&indices, seq_len);

    let input_ids_data: Vec<Vec<u32>> = batch_seqs.iter()
        .map(|s| s[..s.len().min(seq_len)].to_vec())
        .collect();
    let labels_data: Vec<Vec<i64>> = batch_seqs.iter()
        .map(|s| s[1..s.len().min(seq_len + 1)].iter().map(|&id| id as i64).collect())
        .collect();

    let actual_len = input_ids_data[0].len();
    let batch_count = indices.len();

    Ok((
        Tensor::new(input_ids_data, device)?,
        Tensor::ones((batch_count, actual_len), DType::U32, device)?,
        Tensor::new(labels_data, device)?,
    ))
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

/// Extract step number from checkpoint filename.
/// e.g. "checkpoint-50.safetensors" → 50, "checkpoint-emergency-step1234.safetensors" → 1234
fn detect_step_from_path(path: &PathBuf) -> usize {
    let stem = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // Try "checkpoint-{N}" pattern
    if let Some(rest) = stem.strip_prefix("checkpoint-") {
        if let Ok(n) = rest.parse::<usize>() {
            return n;
        }
        // Try "checkpoint-emergency-step{N}"
        if let Some(step_str) = rest.strip_prefix("emergency-step") {
            if let Ok(n) = step_str.parse::<usize>() {
                return n;
            }
        }
    }
    0
}

fn save_log(log: &TrainingLog, path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(log)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn detect_format(path: &PathBuf) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => "jsonl".to_string(),
        Some("json") => "squad".to_string(),
        Some("txt") | Some("text") => "text".to_string(),
        _ => "text".to_string(),
    }
}

/// Resolve "auto" tokenizer to the best default for the model size.
/// - tiny, small: gpt2 (50K vocab, no HF token needed, fast for local tests)
/// - medium+, cloud, MoE: meta-llama/Meta-Llama-3-8B (128K vocab, best encoding efficiency)
fn resolve_tokenizer(name: &str, model_size: &str) -> String {
    if name != "auto" {
        return name.to_string();
    }
    match model_size {
        "tiny" | "small" | "tiny-moe" => "gpt2".to_string(),
        _ => "meta-llama/Meta-Llama-3-8B".to_string(),
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
