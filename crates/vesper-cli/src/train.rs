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

use std::collections::HashMap;
use vesper_core::{VesperConfig, VesperLM, CheckpointData, LayerBoundaries, MappedDataset};
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
    ppl: f32,
}

/// Which optimizer to use
enum OptimizerKind {
    Velvet(VelvetOptimizer),
    AdamW(AdamW),
}

impl OptimizerKind {
    /// AdamW step: uses Candle's full autograd (all layers in one backward).
    /// Only suitable for small batches due to Candle's memory model.
    fn step_adamw(&mut self, loss: &Tensor) -> Result<()> {
        match self {
            OptimizerKind::AdamW(opt) => {
                opt.backward_step(loss)?;
            }
            _ => anyhow::bail!("step_adamw called on non-AdamW optimizer"),
        }
        Ok(())
    }

    /// Layer-by-layer backward: recompute 1 layer at a time during backward.
    ///
    /// Memory: model(2GB) + optimizer(8GB) + 1 layer graph(~1-2GB) + boundaries(~0.5GB)
    ///       = ~12-15GB for batch 10, ~27GB for batch 64.
    /// vs full autograd: ~70GB for batch 10.
    ///
    /// The fused LayerNorm in each layer severs gradient flow through attention/FFN,
    /// so the upstream gradient is constant across all layers (flows only through
    /// residual connections). This means we can use the same upstream for every layer
    /// — mathematically equivalent to Candle's full autograd for this architecture.
    fn step_layerwise(
        &mut self,
        model: &VesperLM,
        boundaries: LayerBoundaries,
        labels: &Tensor,
        input_ids: &Tensor,
        varmap: &VarMap,
    ) -> Result<f32> {
        match self {
            OptimizerKind::Velvet(opt) => {
                let LayerBoundaries { boundaries, mask_4d, num_layers } = boundaries;
                let vocab_size = model.config().vocab_size;
                let aux_weight = model.config().moe_aux_loss_weight;

                // Cache var tensor references for gradient extraction
                let var_info: Vec<(String, Tensor)> = {
                    let data = varmap.data().lock().unwrap();
                    data.iter().map(|(n, v)| (n.clone(), v.as_tensor().clone())).collect()
                };
                let mut named_grads: HashMap<String, Tensor> = HashMap::new();

                // === Phase 0: Head backward (manual layer norm + lm_head) ===
                let head_input_var = candle_core::Var::from_tensor(&boundaries[num_layers])
                    .map_err(|e| anyhow::anyhow!("Var::from_tensor failed: {}", e))?;

                let logits = model.forward_head(head_input_var.as_tensor())
                    .map_err(|e| anyhow::anyhow!("forward_head failed: {}", e))?;
                let loss = compute_loss(&logits, labels)?;
                let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

                // Entropy-adaptive LR from loss value (replaces compute_logits_entropy
                // which wasted ~5GB creating [B, S, 128K] softmax/log/entropy tensors)
                let max_entropy = (vocab_size as f64).ln();
                let approx_entropy = ((loss_val as f64).min(max_entropy) / max_entropy).clamp(0.0, 1.0);
                let entropy_scale = (approx_entropy / 0.5).clamp(0.5, 2.0);
                opt.set_entropy_scale(entropy_scale);
                let current_ppl = (loss_val as f64).exp();
                let ppl_scale = (40.0 / current_ppl.max(1.0)).clamp(0.5, 2.0);
                opt.set_perplexity_scale(ppl_scale);

                let gs = loss.backward()
                    .map_err(|e| anyhow::anyhow!("Head backward failed: {}", e))?;

                // upstream = dL/d(head_input) — constant for all layers (see doc above)
                let upstream = gs.get(head_input_var.as_tensor())
                    .ok_or_else(|| anyhow::anyhow!(
                        "No gradient for head input Var"
                    ))?.clone();

                // Extract head param grads (lm_head, final_norm weights)
                for (name, tensor) in &var_info {
                    if let Some(grad) = gs.get(tensor) {
                        named_grads.insert(name.clone(), grad.clone());
                    }
                }

                // Free head computation graph
                drop(gs);
                drop(logits);
                drop(loss);
                drop(head_input_var);

                // === Phase 1: Layer-by-layer backward (reverse order) ===
                // Each layer: recompute forward → proxy_loss → backward → extract grads → drop
                // Peak memory: only 1 layer's computation graph alive at a time.
                for layer_idx in (0..num_layers).rev() {
                    // Layer 0: recompute from input_ids through embeddings (captures embedding grads)
                    // Layer N>0: use detached boundary as input
                    let input = if layer_idx == 0 {
                        model.embeddings_forward(input_ids)
                            .map_err(|e| anyhow::anyhow!("Embeddings forward failed: {}", e))?
                    } else {
                        boundaries[layer_idx].clone()
                    };

                    let (output, aux) = model.recompute_layer(
                        layer_idx, &input, mask_4d.as_ref(),
                    ).map_err(|e| anyhow::anyhow!("Recompute layer {} failed: {}", layer_idx, e))?;

                    // Proxy loss in F32 for numerical stability (BF16 sum over millions of elements loses precision)
                    let mut proxy_loss = output.to_dtype(DType::F32)?.mul(&upstream.detach().to_dtype(DType::F32)?)?.sum_all()?;
                    if let Some(a) = aux {
                        proxy_loss = proxy_loss.add(&(a * aux_weight)?)?;
                    }

                    let gs = proxy_loss.backward()
                        .map_err(|e| anyhow::anyhow!("Backward layer {} failed: {}", layer_idx, e))?;

                    // Extract param grads for this layer (+ embeddings for layer 0)
                    for (name, tensor) in &var_info {
                        if let Some(grad) = gs.get(tensor) {
                            named_grads.insert(name.clone(), grad.clone());
                        }
                    }

                    // Free this layer's GradStore + computation graph
                    drop(gs);
                }

                // Free boundaries and mask before optimizer step
                drop(boundaries);
                drop(mask_4d);
                drop(upstream);

                // Optimizer step with global gradient clipping
                opt.step_with_named_grads(&named_grads, varmap)
                    .map_err(|e| anyhow::anyhow!("Optimizer step failed: {}", e))?;

                Ok(loss_val)
            }
            OptimizerKind::AdamW(_) => {
                anyhow::bail!(
                    "Layer-wise backward requires the Velvet optimizer. Use --optimizer velvet."
                );
            }
        }
    }

    /// Checkpointed training step: multi-phase backward through segments.
    /// Returns loss_val.
    ///
    /// Phase 0: Build logits from last_hidden via forward_head (manual layer norm with
    ///          gradient flow). backward() gives dL/d(head_input). Then DROP the head
    ///          GradStore to free intermediate gradient tensors (~1GB).
    /// Phase 1: For each segment in reverse, recompute forward + proxy loss backward.
    ///          After each backward, extract ONLY parameter gradients into a HashMap
    ///          and DROP the GradStore (which holds ~2.5GB of intermediate gradients).
    ///
    /// Key optimization: GradStore from backward() holds gradients for ALL computation
    /// graph nodes (attention scores, FFN intermediates, etc.), not just parameters.
    /// By extracting only parameter gradients and dropping the GradStore immediately,
    /// we save ~2.5GB per segment (~15GB total for 6 segments).
    fn step_checkpointed(
        &mut self,
        model: &VesperLM,
        checkpoint_data: CheckpointData,
        labels: &Tensor,
        input_ids: &Tensor,
        varmap: &VarMap,
        vocab_size: usize,
    ) -> Result<f32> {
        match self {
            OptimizerKind::Velvet(opt) => {
                let CheckpointData {
                    boundary_vars, mask_4d, segment_ranges,
                    last_hidden, total_aux_loss: _,
                } = checkpoint_data;

                // Cache var references for gradient extraction (avoids repeated varmap locking)
                let var_info: Vec<(String, Tensor)> = {
                    let data = varmap.data().lock().unwrap();
                    data.iter().map(|(n, v)| (n.clone(), v.as_tensor().clone())).collect()
                };
                let mut named_grads: HashMap<String, Tensor> = HashMap::new();

                // Phase 0: backward through head (manual layer norm + lm_head)
                let head_input_var = candle_core::Var::from_tensor(&last_hidden)
                    .map_err(|e| anyhow::anyhow!("Var::from_tensor failed: {}", e))?;
                drop(last_hidden);

                let logits = model.forward_head(head_input_var.as_tensor())
                    .map_err(|e| anyhow::anyhow!("forward_head failed: {}", e))?;
                let loss = compute_loss(&logits, labels)?;
                let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

                // Entropy-Adaptive LR (from loss, avoids 5GB logits copy)
                let max_entropy = (vocab_size as f64).ln();
                let approx_entropy = ((loss_val as f64).min(max_entropy) / max_entropy).clamp(0.0, 1.0);
                let entropy_scale = (approx_entropy / 0.5).clamp(0.5, 2.0);
                opt.set_entropy_scale(entropy_scale);
                let current_ppl = (loss_val as f64).exp();
                let ppl_scale = (40.0 / current_ppl.max(1.0)).clamp(0.5, 2.0);
                opt.set_perplexity_scale(ppl_scale);

                let grad_store_head = loss.backward()
                    .map_err(|e| anyhow::anyhow!("Backward head failed: {}", e))?;

                let head_upstream = grad_store_head.get(head_input_var.as_tensor())
                    .ok_or_else(|| anyhow::anyhow!(
                        "No gradient for head input Var (forward_head backward didn't reach it)"
                    ))?.clone();

                // Extract ONLY parameter gradients from head (lm_head, final_norm weights)
                for (name, tensor) in &var_info {
                    if let Some(grad) = grad_store_head.get(tensor) {
                        named_grads.insert(name.clone(), grad.clone());
                    }
                }

                // Free head computation graph + all intermediate gradient tensors
                drop(loss);
                drop(logits);
                drop(head_input_var);
                drop(grad_store_head);

                // Phase 1: backward through ALL segments in reverse using proxy loss
                let num_segments = segment_ranges.len();
                let mut upstream = head_upstream;

                for seg_idx in (0..num_segments).rev() {
                    let (recomputed_output, recomputed_aux) = model.recompute_segment(
                        seg_idx, input_ids, &boundary_vars,
                        mask_4d.as_ref(), &segment_ranges,
                    ).map_err(|e| anyhow::anyhow!("Recompute segment {} failed: {}", seg_idx, e))?;

                    // Proxy loss: (output * upstream_grad).sum_all()
                    let mut proxy_loss = recomputed_output.mul(&upstream.detach())?.sum_all()?;

                    if let Some(aux) = recomputed_aux {
                        proxy_loss = proxy_loss.add(
                            &(aux * model.config().moe_aux_loss_weight)?,
                        )?;
                    }

                    let grad_store_seg = proxy_loss.backward()
                        .map_err(|e| anyhow::anyhow!("Backward segment {} failed: {}", seg_idx, e))?;

                    // Extract upstream gradient for the previous segment
                    if seg_idx > 0 {
                        upstream = grad_store_seg.get(boundary_vars[seg_idx].as_tensor())
                            .ok_or_else(|| anyhow::anyhow!(
                                "No gradient for segment {} boundary", seg_idx
                            ))?.clone();
                    }

                    // Extract ONLY parameter gradients (not intermediate gradients)
                    for (name, tensor) in &var_info {
                        if let Some(grad) = grad_store_seg.get(tensor) {
                            named_grads.insert(name.clone(), grad.clone());
                        }
                    }

                    // DROP the full GradStore — frees ~2.5GB of intermediate gradient tensors
                    drop(grad_store_seg);
                }

                // Free boundary vars and mask before optimizer step
                drop(boundary_vars);
                drop(mask_4d);
                drop(upstream);

                // Optimizer step with extracted parameter gradients + global clipping
                opt.step_with_named_grads(&named_grads, varmap)
                    .map_err(|e| anyhow::anyhow!("Optimizer step failed: {}", e))?;

                Ok(loss_val)
            }
            OptimizerKind::AdamW(_) => {
                anyhow::bail!(
                    "Gradient checkpointing is only supported with the Velvet optimizer. \
                     Use --optimizer velvet."
                );
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            OptimizerKind::Velvet(_) => "Velvet",
            OptimizerKind::AdamW(_) => "AdamW",
        }
    }

    /// Format real-time metrics string for progress bar
    fn metrics_str(&self, loss: f32, ppl: f32, step: usize) -> String {
        match self {
            OptimizerKind::Velvet(opt) => {
                format!(
                    "loss: {:.4} | ppl: {:.1} | lr: {:.2e} | \u{03B2}1: {:.3} | gnorm: {:.2} | step: {}",
                    loss, ppl, opt.effective_lr(), opt.effective_beta1(),
                    opt.last_grad_norm(), step,
                )
            }
            OptimizerKind::AdamW(_) => {
                format!("loss: {:.4} | ppl: {:.1} | step: {}", loss, ppl, step)
            }
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
    dtype_str: String,
    gradient_checkpointing: usize,
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
    // Apply gradient checkpointing
    let config = if gradient_checkpointing > 0 {
        config.with_gradient_checkpointing(gradient_checkpointing)
    } else { config };
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
    let dtype = match dtype_str.as_str() {
        "f32" => DType::F32,
        "bf16" | "bfloat16" => DType::BF16,
        "f16" | "float16" => DType::F16,
        _ => anyhow::bail!("Unknown dtype: {}. Use: f32, bf16, f16", dtype_str),
    };
    println!("  Precision: {}", dtype_str);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
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
    if config.gradient_checkpoint_segments > 0 {
        let segs = config.gradient_checkpoint_segments;
        let lps = (config.num_layers + segs - 1) / segs;
        println!("  Gradient checkpointing: {} segments ({} layers each)", segs, lps);
    }

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
    let _vocab_size = model.config().vocab_size;
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
            .template("  {bar:30.green/black} {pos}/{len} [{elapsed}<{eta}] {msg}")
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

            // Forward + backward + optimizer step
            let loss_val: f32;
            let use_layerwise = matches!(optimizer, OptimizerKind::Velvet(_));
            if use_layerwise {
                // Layer-by-layer backward: ~12-15GB for batch 10, ~27GB for batch 64
                let boundaries = model.forward_with_boundaries(&input_ids, Some(&attention_mask))
                    .map_err(|e| anyhow::anyhow!("Forward failed: {}", e))?;
                loss_val = optimizer.step_layerwise(
                    model, boundaries, &labels, &input_ids, varmap,
                )?;
            } else {
                // AdamW: full autograd (high memory, small batches only)
                let (logits, aux_loss_opt) = model.forward(&input_ids, Some(&attention_mask))?;
                let mut loss = compute_loss(&logits, &labels)?;
                if let Some(aux_loss) = aux_loss_opt {
                    loss = loss.add(&(aux_loss * model.config().moe_aux_loss_weight)?)?;
                }
                loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                optimizer.step_adamw(&loss)?;
            }

            epoch_loss += loss_val;
            count += 1;
            global_step += 1;

            let ppl_val = (loss_val as f64).exp() as f32;
            log.steps.push(StepLog { step: global_step, loss: loss_val, ppl: ppl_val });

            let avg_loss = epoch_loss / count as f32;
            let avg_ppl = (avg_loss as f64).exp() as f32;
            pb.set_message(optimizer.metrics_str(avg_loss, avg_ppl, global_step));
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
    let _vocab_size = model.config().vocab_size;
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

                let loss_val: f32;
                let use_layerwise = matches!(optimizer, OptimizerKind::Velvet(_));
                if use_layerwise {
                    let boundaries = model.forward_with_boundaries(&input_ids, Some(&attention_mask))
                        .map_err(|e| anyhow::anyhow!("Forward failed: {}", e))?;
                    loss_val = optimizer.step_layerwise(
                        model, boundaries, &labels, &input_ids, varmap,
                    )?;
                } else {
                    let (logits, aux_loss_opt) = model.forward(&input_ids, Some(&attention_mask))?;
                    let mut loss = compute_loss(&logits, &labels)?;
                    if let Some(aux_loss) = aux_loss_opt {
                        loss = loss.add(&(aux_loss * model.config().moe_aux_loss_weight)?)?;
                    }
                    loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                    optimizer.step_adamw(&loss)?;
                }

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
    // F32 for numerical stability: BF16 softmax over 128K vocab causes NaN
    let logits_2d = logits.to_dtype(DType::F32)?.reshape((batch_size * seq_len, vocab_size))?;
    let labels_1d = labels.flatten_all()?;
    cross_entropy(&logits_2d, &labels_1d)
        .map_err(|e| anyhow::anyhow!("Cross entropy failed: {}", e))
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
