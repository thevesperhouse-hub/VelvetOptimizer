//! tch-rs Training Pipeline — PyTorch-level memory efficiency
//!
//! Standard autograd: forward → loss → backward → step.
//! PyTorch frees intermediate activations during backward automatically.
//! No manual layer-by-layer tricks needed.
//!
//! Build: cargo build --release -p vesper-cli --features tch-backend
//! Requires: LIBTORCH_USE_PYTORCH=1 (pip install torch) or LIBTORCH path

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

use vesper_core::VesperConfig;
use vesper_training::{DatasetLoader, StreamingTextLoader};

use crate::tch_model::TchVesperLM;
use crate::tokenizer;

// ======================== Training Log ========================

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
    ppl: f32,
}

// ======================== Velvet Optimizer (tch) ========================

struct TchVelvetOptimizer {
    config: vesper_optimizer::VelvetConfig,
    step: usize,
    entropy_scale: f64,
    perplexity_scale: f64,
    last_grad_norm: f64,
    states: HashMap<String, (Tensor, Tensor, usize)>, // (m, v, local_step)
}

impl TchVelvetOptimizer {
    fn new(config: vesper_optimizer::VelvetConfig) -> Self {
        Self {
            config,
            step: 0,
            entropy_scale: 1.0,
            perplexity_scale: 1.0,
            last_grad_norm: 0.0,
            states: HashMap::new(),
        }
    }

    /// Forward + backward + optimizer step. Returns loss value.
    fn backward_step(&mut self, loss: &Tensor, vs: &mut nn::VarStore) -> f32 {
        let loss_val = loss.double_value(&[]) as f32;

        self.step += 1;

        // Adaptive LR from loss
        let max_entropy = (self.config.lr * 1000.0).max(10.0).ln(); // rough proxy
        let approx_entropy = ((loss_val as f64).min(max_entropy) / max_entropy).clamp(0.0, 1.0);
        self.entropy_scale = (approx_entropy / 0.5).clamp(0.5, 2.0);
        let current_ppl = (loss_val as f64).exp();
        self.perplexity_scale = (40.0 / current_ppl.max(1.0)).clamp(0.5, 2.0);

        // Backward
        vs.zero_grad();
        loss.backward();

        // Effective LR and beta1
        let effective_lr = if self.config.entropy_adaptive {
            self.config.lr * self.entropy_scale
        } else {
            self.config.lr
        };
        let effective_beta1 = if self.config.perplexity_guided {
            (self.config.beta1 * self.perplexity_scale).clamp(0.5, 0.999)
        } else {
            self.config.beta1
        };

        // Gradient clipping (global norm)
        let named_vars = vs.variables();
        if self.config.max_grad_norm > 0.0 {
            let mut total_norm_sq = 0.0f64;
            for (_name, var) in &named_vars {
                let grad = var.grad();
                if grad.defined() {
                    total_norm_sq += grad.square().sum(Kind::Float).double_value(&[]);
                }
            }
            let global_norm = total_norm_sq.sqrt();
            self.last_grad_norm = global_norm;

            if global_norm > self.config.max_grad_norm {
                let clip_coef = self.config.max_grad_norm / (global_norm + 1e-6);
                tch::no_grad(|| {
                    for (_name, var) in &named_vars {
                        let grad = var.grad();
                        if grad.defined() {
                            let _ = grad.mul_scalar_(clip_coef);
                        }
                    }
                });
            }
        }

        // Apply Velvet update (AdamW + adaptive features)
        tch::no_grad(|| {
            for (name, var) in &named_vars {
                let grad = var.grad();
                if !grad.defined() { continue; }

                // Ensure F32 for optimizer math
                let grad_f32 = grad.to_kind(Kind::Float);
                let var_f32 = var.to_kind(Kind::Float);

                // Get or init state
                let state = self.states.entry(name.clone()).or_insert_with(|| {
                    let m = Tensor::zeros_like(&var_f32);
                    let v = Tensor::zeros_like(&var_f32);
                    (m, v, 0)
                });
                state.2 += 1;

                // Bias correction
                let bc1 = 1.0 - effective_beta1.powi(state.2 as i32);
                let bc2 = 1.0 - self.config.beta2.powi(state.2 as i32);

                // Decoupled weight decay
                let _ = var.add_(&(&var_f32 * (-effective_lr * self.config.weight_decay)));

                // Update moments
                state.0 = &state.0 * effective_beta1 + &grad_f32 * (1.0 - effective_beta1);
                state.1 = &state.1 * self.config.beta2 + grad_f32.square() * (1.0 - self.config.beta2);

                // Bias-corrected estimates
                let m_hat = &state.0 / bc1;
                let v_hat = &state.1 / bc2;

                // Parameter update
                let update = &m_hat / (v_hat.sqrt() + self.config.eps);
                let _ = var.add_(&(update * (-effective_lr)).to_kind(var.kind()));
            }
        });

        loss_val
    }

    fn effective_lr(&self) -> f64 {
        if self.config.entropy_adaptive {
            self.config.lr * self.entropy_scale
        } else {
            self.config.lr
        }
    }

    fn effective_beta1(&self) -> f64 {
        if self.config.perplexity_guided {
            (self.config.beta1 * self.perplexity_scale).clamp(0.5, 0.999)
        } else {
            self.config.beta1
        }
    }
}

// ======================== Optimizer Wrapper ========================

enum TchOptimizer {
    Velvet(TchVelvetOptimizer),
    AdamW(nn::Optimizer),
}

impl TchOptimizer {
    fn backward_step(&mut self, loss: &Tensor, vs: &mut nn::VarStore) -> f32 {
        match self {
            TchOptimizer::Velvet(opt) => opt.backward_step(loss, vs),
            TchOptimizer::AdamW(opt) => {
                let loss_val = loss.double_value(&[]) as f32;
                opt.backward_step(loss);
                loss_val
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            TchOptimizer::Velvet(_) => "Velvet",
            TchOptimizer::AdamW(_) => "AdamW",
        }
    }

    fn metrics_str(&self, loss: f32, ppl: f32, step: usize) -> String {
        match self {
            TchOptimizer::Velvet(opt) => {
                format!(
                    "loss: {:.4} | ppl: {:.1} | lr: {:.2e} | β1: {:.3} | gnorm: {:.2} | step: {}",
                    loss, ppl, opt.effective_lr(), opt.effective_beta1(),
                    opt.last_grad_norm, step,
                )
            }
            TchOptimizer::AdamW(_) => {
                format!("loss: {:.4} | ppl: {:.1} | step: {}", loss, ppl, step)
            }
        }
    }
}

// ======================== Loss Computation ========================

fn compute_loss(logits: &Tensor, labels: &Tensor) -> Tensor {
    let dims = logits.size();
    let (batch, seq, vocab) = (dims[0], dims[1], dims[2]);
    // F32 for numerical stability (BF16 softmax over 128K vocab → NaN)
    let logits_2d = logits.reshape(&[batch * seq, vocab]).to_kind(Kind::Float);
    let labels_1d = labels.reshape(&[batch * seq]);
    // log_softmax + gather = numerically stable cross-entropy
    let log_probs = logits_2d.log_softmax(-1, Kind::Float);
    let nll = -log_probs.gather(1, &labels_1d.unsqueeze(1), false).squeeze_dim(1);
    nll.mean(Kind::Float)
}

// ======================== Batch Preparation ========================

fn prepare_batch_inmemory(
    ds: &vesper_training::Dataset,
    start: usize,
    end: usize,
    device: Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut all_input_ids: Vec<i64> = Vec::new();
    let mut all_masks: Vec<i64> = Vec::new();
    let mut all_labels: Vec<i64> = Vec::new();
    let mut count = 0i64;
    let mut seq_len = 0i64;

    for idx in start..end {
        if let Some(s) = ds.get(idx) {
            seq_len = s.input_ids.len() as i64;
            all_input_ids.extend(s.input_ids.iter().map(|&x| x as i64));
            all_masks.extend(s.attention_mask.iter().map(|&x| x as i64));
            all_labels.extend(s.labels.iter().map(|&x| x));
            count += 1;
        }
    }
    if count == 0 {
        anyhow::bail!("Empty batch");
    }

    let input_ids = Tensor::from_slice(&all_input_ids).view([count, seq_len]).to_device(device);
    let masks = Tensor::from_slice(&all_masks).view([count, seq_len]).to_device(device);
    let labels = Tensor::from_slice(&all_labels).view([count, seq_len]).to_device(device);

    Ok((input_ids, masks, labels))
}

fn prepare_batch_cached(
    mapped: &vesper_core::MappedDataset,
    start: usize,
    end: usize,
    seq_len: usize,
    device: Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let indices: Vec<usize> = (start..end).collect();
    let batch_seqs = mapped.get_batch(&indices, seq_len);
    let batch_count = indices.len() as i64;

    let mut all_input_ids: Vec<i64> = Vec::new();
    let mut all_labels: Vec<i64> = Vec::new();
    let mut actual_len = 0i64;

    for s in &batch_seqs {
        let len = s.len().min(seq_len);
        actual_len = len as i64;
        all_input_ids.extend(s[..len].iter().map(|&x| x as i64));
        all_labels.extend(s[1..len.min(seq_len + 1).min(s.len())].iter().map(|&x| x as i64));
    }

    // Pad labels to same length as input_ids if needed
    let label_len = all_labels.len() as i64 / batch_count;
    if label_len < actual_len {
        // Labels are 1 shorter than inputs (shift), pad the last position
        let mut padded_labels: Vec<i64> = Vec::new();
        for s in &batch_seqs {
            let len = s.len().min(seq_len);
            let labels: Vec<i64> = s[1..len].iter().map(|&x| x as i64).collect();
            padded_labels.extend(&labels);
            // Pad remaining with 0
            for _ in labels.len()..actual_len as usize {
                padded_labels.push(0);
            }
        }
        all_labels = padded_labels;
    }

    let input_ids = Tensor::from_slice(&all_input_ids).view([batch_count, actual_len]).to_device(device);
    let masks = Tensor::ones(&[batch_count, actual_len], (Kind::Int64, device));
    let labels = Tensor::from_slice(&all_labels).view([batch_count, actual_len]).to_device(device);

    Ok((input_ids, masks, labels))
}

// ======================== Main Entry Point ========================

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
    dtype_str: String,
) -> Result<()> {
    println!("\n=== VesperAI Training (tch-rs / PyTorch backend) ===\n");

    // Setup SIGTERM/SIGINT handler
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        eprintln!("\n  SIGTERM/SIGINT received! Saving checkpoint before exit...");
        shutdown_clone.store(true, Ordering::SeqCst);
    }).ok();

    // 1. Device
    let device = if tch::Cuda::is_available() {
        let cuda_count = tch::Cuda::device_count();
        println!("  CUDA available ({} device(s)), using GPU 0", cuda_count);
        Device::Cuda(0)
    } else {
        println!("  No CUDA, using CPU");
        Device::Cpu
    };

    // 2. Model config
    let config = match model_size.as_str() {
        "tiny" => VesperConfig::tiny(),
        "small" => VesperConfig::small(),
        "medium" => VesperConfig::medium(),
        "tiny-moe" => VesperConfig::tiny_moe(),
        "medium-moe" => VesperConfig::medium_moe(),
        "large" => VesperConfig::large(),
        "large-moe" => VesperConfig::large_moe(),
        "xlarge" | "1b" => VesperConfig::xlarge(),
        _ => anyhow::bail!("Unknown model size: {}. Use: tiny, small, medium, large, xlarge", model_size),
    };
    let config = if moe { config.with_moe(num_experts, top_k) } else { config };

    // 3. Tokenizer
    let resolved_tokenizer = resolve_tokenizer(&tokenizer_name, &model_size);
    let tok = tokenizer::load_tokenizer(&resolved_tokenizer)?;
    let vocab_size = if vocab_size_override > 0 { vocab_size_override } else { tokenizer::vocab_size(&tok) };
    let config = config.with_vocab_size(vocab_size);
    config.validate()?;

    let moe_str = if config.moe_enabled {
        format!(" [MoE: {}x top-{} experts]", config.moe_num_experts, config.moe_top_k)
    } else { String::new() };
    println!("  Model: {} ({} params){}", model_size, format_params(config.total_params()), moe_str);
    println!("  Vocab size: {}", vocab_size);

    // 4. Create VarStore + model
    let mut vs = nn::VarStore::new(device);
    let model = TchVesperLM::new(&vs.root(), &config);

    // Cast to requested dtype
    match dtype_str.as_str() {
        "bf16" | "bfloat16" => { vs.bfloat16(); println!("  Precision: BF16"); }
        "f16" | "float16" => { vs.half(); println!("  Precision: F16"); }
        "f32" => { println!("  Precision: F32"); }
        _ => anyhow::bail!("Unknown dtype: {}. Use: f32, bf16, f16", dtype_str),
    }

    // Resume from checkpoint
    let start_step = if let Some(ref ckpt) = resume {
        println!("  Resuming from checkpoint: {}", ckpt.display());
        vs.load(ckpt).context("Failed to load checkpoint")?;
        let step = detect_step_from_path(ckpt);
        println!("  Weights loaded (resuming from step {})", step);
        step
    } else {
        println!("  Model initialized from scratch");
        0
    };
    println!("  Device: {:?}", device);

    // Print VRAM info if CUDA
    if tch::Cuda::is_available() {
        // Parameter memory estimate
        let param_count: i64 = vs.trainable_variables().iter()
            .map(|t| t.size().iter().product::<i64>())
            .sum();
        let bytes_per_param: i64 = match dtype_str.as_str() {
            "bf16" | "f16" | "bfloat16" | "float16" => 2,
            _ => 4,
        };
        let param_mb = param_count * bytes_per_param / (1024 * 1024);
        println!("  Parameters: {} ({} MB)", param_count, param_mb);
    }

    // 5. Optimizer
    let mut optimizer = match optimizer_name.to_lowercase().as_str() {
        "velvet" => {
            let velvet_config = vesper_optimizer::VelvetConfig {
                lr,
                ..vesper_optimizer::VelvetConfig::optimal()
            };
            TchOptimizer::Velvet(TchVelvetOptimizer::new(velvet_config))
        }
        "adamw" => {
            let opt = nn::AdamW::default()
                .beta1(0.9)
                .beta2(0.999)
                .wd(0.01)
                .build(&vs, lr)
                .context("Failed to create AdamW optimizer")?;
            TchOptimizer::AdamW(opt)
        }
        _ => anyhow::bail!("Unknown optimizer: {}. Use: velvet, adamw", optimizer_name),
    };
    println!("  Optimizer: {}", optimizer.name());

    // 6. Load dataset and train
    std::fs::create_dir_all(&output_dir)?;

    if let Some(ref cache) = cache_path {
        println!("\n  Mode: Binary cache");
        let mapped = vesper_core::MappedDataset::load(cache)
            .context("Failed to load binary cache")?;
        println!("  Cache: {} sequences, {} tokens", mapped.num_sequences(), mapped.total_tokens());

        run_training_loop(
            &model, &mut vs, &mut optimizer, device,
            epochs, batch_size, seq_len, save_every, &output_dir, max_steps,
            &shutdown, start_step, &optimizer_name, &model_size,
            TrainingData::Cached(&mapped),
        )?;
    } else if streaming {
        println!("\n  Mode: Streaming ({}MB chunks)", chunk_mb);
        let mut loader = StreamingTextLoader::new(&dataset, tok, seq_len, chunk_mb)?;

        run_streaming_loop(
            &model, &mut vs, &mut optimizer, &mut loader, device,
            epochs, batch_size, save_every, &output_dir, max_steps,
            &shutdown, start_step, &optimizer_name, &model_size,
        )?;
    } else {
        println!("\n  Mode: In-memory");
        let loader = DatasetLoader::from_tokenizer(tok, seq_len);
        let detected_format = if format == "auto" { detect_format(&dataset) } else { format.clone() };
        println!("  Loading dataset: {} (format: {})", dataset.display(), detected_format);
        let start_load = Instant::now();

        let ds = match detected_format.as_str() {
            "text" | "txt" => loader.load_text(&dataset).context("Failed to load text dataset")?,
            "jsonl" => loader.load_jsonl(&dataset).context("Failed to load JSONL dataset")?,
            "squad" | "json" => loader.load_squad(&dataset).context("Failed to load SQuAD dataset")?,
            _ => anyhow::bail!("Unknown format: {}. Use: text, jsonl, squad, auto", detected_format),
        };
        println!("  Dataset loaded in {:.1}s ({} samples)", start_load.elapsed().as_secs_f64(), ds.len());

        run_training_loop(
            &model, &mut vs, &mut optimizer, device,
            epochs, batch_size, seq_len, save_every, &output_dir, max_steps,
            &shutdown, start_step, &optimizer_name, &model_size,
            TrainingData::InMemory(&ds),
        )?;
    }

    Ok(())
}

// ======================== Training Data ========================

enum TrainingData<'a> {
    InMemory(&'a vesper_training::Dataset),
    Cached(&'a vesper_core::MappedDataset),
}

// ======================== Training Loop ========================

fn run_training_loop(
    model: &TchVesperLM,
    vs: &mut nn::VarStore,
    optimizer: &mut TchOptimizer,
    device: Device,
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
    let num_samples = match &data {
        TrainingData::InMemory(ds) => ds.len(),
        TrainingData::Cached(m) => m.num_sequences(),
    };
    let num_batches = (num_samples + batch_size - 1) / batch_size;
    let mut global_step = start_step;
    let mut log = TrainingLog {
        optimizer: optimizer_name.to_string(),
        model_size: model_size.to_string(),
        steps: Vec::new(),
    };
    let log_path = format!("{}/training_log.json", output_dir.display());

    // Resume log
    if start_step > 0 {
        if let Ok(data) = std::fs::read_to_string(&log_path) {
            if let Ok(existing) = serde_json::from_str::<TrainingLog>(&data) {
                log.steps = existing.steps;
                println!("  Resumed training log ({} entries)", log.steps.len());
            }
        }
    }

    let start = Instant::now();
    println!("  Samples: {}, Batches/epoch: {}, Start step: {}\n", num_samples, num_batches, start_step);

    for epoch in 0..epochs {
        println!("[Epoch {}/{}]", epoch + 1, epochs);
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("  {bar:30.green/black} {pos}/{len} [{elapsed}<{eta}] {msg}")
            .unwrap());

        let mut epoch_loss = 0.0f32;
        let mut count = 0usize;

        for batch_idx in 0..num_batches {
            if shutdown.load(Ordering::SeqCst) {
                let path = format!("{}/checkpoint-emergency-step{}.ot", output_dir.display(), global_step);
                vs.save(&path)?;
                save_log(&log, &log_path)?;
                println!("\n  Emergency checkpoint saved: {}", path);
                return Ok(());
            }

            if max_steps > 0 && global_step >= max_steps { break; }

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
            let (logits, aux_loss_opt) = model.forward(&input_ids, Some(&attention_mask));
            let mut loss = compute_loss(&logits, &labels);
            if let Some(aux_loss) = aux_loss_opt {
                loss = &loss + aux_loss * model.config().moe_aux_loss_weight;
            }

            // Backward + step
            let loss_val = optimizer.backward_step(&loss, vs);
            drop(loss);
            drop(logits);

            epoch_loss += loss_val;
            count += 1;
            global_step += 1;

            let ppl_val = (loss_val as f64).exp() as f32;
            log.steps.push(StepLog { step: global_step, loss: loss_val, ppl: ppl_val });

            let avg_loss = epoch_loss / count as f32;
            let avg_ppl = (avg_loss as f64).exp() as f32;
            pb.set_message(optimizer.metrics_str(avg_loss, avg_ppl, global_step));
            pb.inc(1);

            // Periodic checkpoint
            if save_every > 0 && global_step % save_every == 0 {
                let path = format!("{}/checkpoint-{}.ot", output_dir.display(), global_step);
                vs.save(&path)?;
                save_log(&log, &log_path)?;
                pb.println(format!("  Checkpoint saved: {} (step {})", path, global_step));
            }
        }

        let avg = epoch_loss / count.max(1) as f32;
        let ppl = (avg as f64).exp() as f32;
        pb.finish_and_clear();
        println!("  Epoch {}/{}: loss={:.4} ppl={:.2} ({} steps)\n", epoch + 1, epochs, avg, ppl, count);

        // Epoch checkpoint
        let path = format!("{}/checkpoint-epoch{}.ot", output_dir.display(), epoch + 1);
        vs.save(&path)?;
        save_log(&log, &log_path)?;
        println!("  Epoch checkpoint: {}", path);

        if max_steps > 0 && global_step >= max_steps {
            println!("  Reached max_steps ({}), stopping.", max_steps);
            break;
        }
    }

    let final_path = format!("{}/checkpoint-final.ot", output_dir.display());
    vs.save(&final_path)?;
    save_log(&log, &log_path)?;
    println!("\n=== Training Complete ===");
    println!("  Final checkpoint: {}", final_path);
    println!("  Training log: {}", log_path);
    println!("  Total steps: {}", global_step);
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

// ======================== Streaming Training Loop ========================

fn run_streaming_loop(
    model: &TchVesperLM,
    vs: &mut nn::VarStore,
    optimizer: &mut TchOptimizer,
    loader: &mut StreamingTextLoader,
    device: Device,
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
    let mut global_step = start_step;
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
        let mut epoch_loss = 0.0f32;
        let mut epoch_count = 0usize;
        let mut chunk_idx = 0;

        while let Some(chunk_dataset) = loader.next_chunk()? {
            chunk_idx += 1;
            let num_batches = (chunk_dataset.len() + batch_size - 1) / batch_size;

            let pb = ProgressBar::new(num_batches as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template(&format!("  chunk {} [{{bar:30.green/black}}] {{pos}}/{{len}} [{{elapsed}}] {{msg}}", chunk_idx))
                .unwrap());

            for batch_idx in 0..num_batches {
                if shutdown.load(Ordering::SeqCst) {
                    let path = format!("{}/checkpoint-emergency-step{}.ot", output_dir.display(), global_step);
                    vs.save(&path)?;
                    save_log(&log, &log_path)?;
                    println!("\n  Emergency checkpoint saved: {}", path);
                    return Ok(());
                }

                if max_steps > 0 && global_step >= max_steps { break; }

                let s = batch_idx * batch_size;
                let e = (s + batch_size).min(chunk_dataset.len());
                let (input_ids, attention_mask, labels) =
                    prepare_batch_inmemory(&chunk_dataset, s, e, device)?;

                let (logits, aux_loss_opt) = model.forward(&input_ids, Some(&attention_mask));
                let mut loss = compute_loss(&logits, &labels);
                if let Some(aux_loss) = aux_loss_opt {
                    loss = &loss + aux_loss * model.config().moe_aux_loss_weight;
                }

                let loss_val = optimizer.backward_step(&loss, vs);
                drop(loss);
                drop(logits);

                epoch_loss += loss_val;
                epoch_count += 1;
                global_step += 1;

                let ppl_val = (loss_val as f64).exp() as f32;
                log.steps.push(StepLog { step: global_step, loss: loss_val, ppl: ppl_val });

                let avg_loss = epoch_loss / epoch_count as f32;
                let avg_ppl = (avg_loss as f64).exp() as f32;
                pb.set_message(optimizer.metrics_str(avg_loss, avg_ppl, global_step));
                pb.inc(1);

                if save_every > 0 && global_step % save_every == 0 {
                    let path = format!("{}/checkpoint-{}.ot", output_dir.display(), global_step);
                    vs.save(&path)?;
                    save_log(&log, &log_path)?;
                    pb.println(format!("  Checkpoint saved: {}", path));
                }
            }
            pb.finish_and_clear();
            if max_steps > 0 && global_step >= max_steps { break; }
        }

        let avg = epoch_loss / epoch_count.max(1) as f32;
        let ppl = (avg as f64).exp() as f32;
        println!("  Epoch {}/{}: loss={:.4} ppl={:.2} ({} chunks, {} steps)",
            epoch + 1, epochs, avg, ppl, chunk_idx, epoch_count);

        let path = format!("{}/checkpoint-epoch{}.ot", output_dir.display(), epoch + 1);
        vs.save(&path)?;
        save_log(&log, &log_path)?;
        println!("  Epoch checkpoint: {}", path);

        if max_steps > 0 && global_step >= max_steps {
            println!("  Reached max_steps ({}), stopping.", max_steps);
            break;
        }
    }

    let final_path = format!("{}/checkpoint-final.ot", output_dir.display());
    vs.save(&final_path)?;
    save_log(&log, &log_path)?;
    println!("\n=== Training Complete ===");
    println!("  Final checkpoint: {}", final_path);
    println!("  Total steps: {}", global_step);
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

// ======================== Helpers ========================

fn detect_step_from_path(path: &PathBuf) -> usize {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if let Some(rest) = stem.strip_prefix("checkpoint-") {
        if let Ok(n) = rest.parse::<usize>() { return n; }
        if let Some(step_str) = rest.strip_prefix("emergency-step") {
            if let Ok(n) = step_str.parse::<usize>() { return n; }
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

fn resolve_tokenizer(name: &str, model_size: &str) -> String {
    if name != "auto" { return name.to_string(); }
    match model_size {
        "tiny" | "small" | "tiny-moe" => "gpt2".to_string(),
        _ => "meta-llama/Meta-Llama-3-8B".to_string(),
    }
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 { format!("{:.1}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.0}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.0}K", n as f64 / 1e3) }
    else { format!("{}", n) }
}
