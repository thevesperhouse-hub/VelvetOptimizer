# VelvetOptimizer

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.83+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

> High-performance LLM training framework with a custom optimizer, Mixture of Experts, and CUDA/Triton kernels. Available in **Rust** (Candle/tch-rs) and **Python** (PyTorch).

VelvetOptimizer is a from-scratch transformer training system built around the **Velvet optimizer** — an enhanced AdamW with entropy-adaptive learning rate, perplexity-guided momentum, and custom CUDA kernels. It includes **VesperLM**, a transformer architecture with FlyLoRA (sparse low-rank adaptation), ERA activation, and Mixture of Experts (MoE).

### Training Backends

| Backend | Language | Status | Best For |
|---------|----------|--------|----------|
| **Python/PyTorch** | Python | Production | Large-scale GPU training (recommended) |
| **tch-rs** | Rust | Production | Rust-native training with PyTorch autograd |
| **Candle** | Rust | Development | CPU/small models, inference |

> **Why multiple backends?** Candle's autograd retains ALL intermediate tensors until backward completes (~70GB VRAM for batch 10 on a 1B model). PyTorch frees intermediates progressively during backward, enabling batch 64+ on the same hardware. The Python and tch-rs backends use PyTorch's autograd for memory-efficient GPU training.

---

## Features

- **Velvet Optimizer** — AdamW + entropy-adaptive LR + perplexity-guided momentum + custom CUDA/Triton kernels
- **VesperLM** — Transformer with multi-head attention, RoPE, FlyLoRA, ERA activation
- **Mixture of Experts (MoE)** — Top-K routing, N expert FFNs per layer, Switch Transformer load balancing loss
- **Flash Attention 2** — via PyTorch SDPA (Python) or manual implementation (Rust)
- **Triton Kernels** — Fused Velvet update, fused cross-entropy (Unsloth-style), fused ERA activation
- **Gradient Checkpointing** — `torch.utils.checkpoint` (Python) or manual segment-based (Rust)
- **Gradient Accumulation** — `--grad-accum N` for larger effective batch sizes
- **CLI Training** — `vesper train` (Rust) or `python train.py` (Python) with checkpointing, resume, SIGTERM auto-save
- **Streaming** — Chunk-by-chunk training for large datasets (>1GB), with multi-worker sharding
- **Binary Cache** — Memory-mapped dataset cache for instant loading
- **Wandb Integration** — `--wandb` for Weights & Biases experiment tracking (Python)
- **Evaluation Loop** — `--val-dataset` for periodic validation (Python)
- **Benchmarking** — `vesper benchmark` for Velvet vs AdamW comparison
- **Text Generation** — `vesper generate` with temperature/top-p sampling
- **Cloud Ready** — Dockerfile + Vast.ai setup script for H100/A100 training

---

## Quick Start

### Prerequisites

- Rust 1.83+
- CUDA Toolkit 12.8 (with `nvcc` in PATH)
- Visual Studio Build Tools 2022 (MSVC v143) on Windows

### Build

```bash
git clone https://github.com/thevesperhouse-hub/VelvetOptimizer.git
cd VelvetOptimizer
cargo build --release -p vesper-cli
```

### Download a dataset

```bash
pip install datasets
python scripts/download_fineweb.py --tokens 10M --output fineweb-10M.txt
```

### Train

```bash
# Standard training (Velvet optimizer)
./target/release/vesper train \
  --dataset fineweb-10M.txt \
  --tokenizer gpt2 \
  --model-size small \
  --epochs 3 \
  --batch-size 4 \
  --lr 3e-4

# MoE training
./target/release/vesper train \
  --dataset fineweb-10M.txt \
  --tokenizer gpt2 \
  --model-size tiny-moe \
  --epochs 3

# With explicit MoE flags
./target/release/vesper train \
  --dataset fineweb-10M.txt \
  --tokenizer gpt2 \
  --model-size small \
  --moe --num-experts 8 --top-k 2 \
  --epochs 3
```

### Python Training (Recommended for GPU)

```bash
cd python/
pip install -r requirements.txt    # PyTorch CUDA 12.8 + Triton + Flash Attn

# Basic training
python train.py --dataset data.txt --model-size medium --dtype bf16

# Large-scale (1B params, H100)
python train.py --dataset data.txt --model-size xlarge --dtype bf16 \
  --gradient-checkpointing --batch-size 16 --grad-accum 4 \
  --wandb --wandb-project vesper-1B

# MoE with validation
python train.py --dataset train.txt --val-dataset val.txt \
  --model-size large --moe --num-experts 8 --top-k 2 --eval-every 500

# Resume from checkpoint
python train.py --dataset data.txt --model-size xlarge --resume checkpoints/step_5000.pt
```

### tch-rs Backend (Rust + PyTorch autograd)

```bash
# Server setup (Linux with PyTorch installed via pip)
pip install torch
export LIBTORCH_USE_PYTORCH=1
cargo build --release -p vesper-cli --features tch-backend

# Train with tch-rs backend
./target/release/vesper train --backend tch --dataset data.txt --model-size xlarge --dtype bf16
```

### Benchmark (Velvet vs AdamW)

```bash
./target/release/vesper benchmark \
  --dataset fineweb-10M.txt \
  --tokenizer gpt2 \
  --model-size small \
  --epochs 5

# Plot results
python scripts/plot_benchmark.py benchmark_report.json
```

### Generate text

```bash
./target/release/vesper generate \
  --model checkpoints/checkpoint-final.safetensors \
  --tokenizer gpt2 \
  --model-size small \
  --prompt "The future of AI" \
  --max-tokens 100 \
  --temperature 0.8
```

---

## Model Sizes

| Size | Layers | Heads | Hidden | Params | Notes |
|------|--------|-------|--------|--------|-------|
| `tiny` | 6 | 4 | 256 | ~17M | Quick local tests |
| `small` | 8 | 8 | 512 | ~35M | Local training |
| `medium` | 12 | 12 | 768 | ~89M | Good balance |
| `large` | 24 | 16 | 1024 | ~350M | Cloud (A100) |
| `xlarge` | 24 | 16 | 2048 | ~1B | Cloud (A100 80GB) |

### MoE Presets

| Size | Experts | Top-K | Base | Total Params |
|------|---------|-------|------|-------------|
| `tiny-moe` | 4 | 2 | tiny | ~26M |
| `medium-moe` | 8 | 2 | medium | ~500M |
| `large-moe` | 16 | 2 | large | ~5B |

MoE multiplies FFN parameters by the number of experts, but only activates K experts per token, so compute scales as K/N of total params.

---

## Architecture

```
VelvetOptimizer/
├── python/                    # ★ Python training framework (recommended)
│   ├── train.py               # CLI training script (argparse)
│   ├── requirements.txt       # PyTorch CUDA 12.8 + Triton 3.6
│   ├── setup.py               # pip install -e .
│   └── vesper/
│       ├── model.py           # VesperLM (Flash Attn 2, RoPE, FlyLoRA, MoE)
│       ├── optimizer.py       # VelvetOptimizer (torch.optim.Optimizer)
│       ├── config.py          # Model configs + presets
│       ├── data.py            # Dataset (inmemory, streaming, cached, JSONL)
│       └── kernels/
│           ├── velvet_triton.py   # Fused optimizer update (Triton)
│           ├── era_triton.py      # Fused ERA activation (Triton)
│           ├── fused_ce.py        # Fused cross-entropy — Unsloth-style (Triton)
│           └── velvet_cuda.py     # Native CUDA kernel (cpp_extension fallback)
│
├── crates/
│   ├── vesper-core/           # Model architecture (Candle)
│   │   ├── model.rs           # VesperLM transformer
│   │   ├── attention.rs       # Multi-head attention + RoPE
│   │   ├── flylora.rs         # Sparse LoRA (75% param reduction)
│   │   ├── era.rs             # ERA: GELU(x) + γ*softplus(x)
│   │   ├── moe.rs             # Mixture of Experts (router + N experts)
│   │   ├── config.rs          # Model configs + presets
│   │   └── dataset_cache.rs   # Memory-mapped binary cache
│   │
│   ├── vesper-optimizer/      # Velvet optimizer
│   │   ├── velvet.rs          # Core optimizer (AdamW + adaptive features)
│   │   └── cuda/kernels.cu    # Custom CUDA kernels
│   │
│   ├── vesper-training/       # Training infrastructure
│   │   ├── trainer.rs         # Training loop
│   │   └── dataset.rs         # Dataset loading + StreamingTextLoader
│   │
│   ├── vesper-cli/            # Command-line interface
│   │   ├── main.rs            # CLI entry (train, benchmark, generate, cache)
│   │   ├── train.rs           # Candle training pipeline
│   │   ├── tch_model.rs       # ★ VesperLM in tch-rs (PyTorch autograd)
│   │   ├── tch_train.rs       # ★ tch-rs training pipeline
│   │   ├── benchmark.rs       # Velvet vs AdamW comparison
│   │   ├── generate.rs        # Text generation
│   │   └── cache.rs           # Dataset cache build/info
│   │
│   ├── vesper-app/            # Tauri desktop application
│   └── vesper-metacog/        # Metacognition module
│
├── scripts/
│   ├── download_fineweb.py    # FineWeb-Edu dataset downloader
│   ├── plot_benchmark.py      # Benchmark visualization (4 graphs)
│   ├── plot_training.py       # Training loss/perplexity curves
│   └── vastai_setup.sh        # One-command Vast.ai setup
│
└── Dockerfile                 # Multi-stage CUDA build for cloud
```

---

## CLI Reference

### `vesper train`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | Path to dataset (.txt, .jsonl, .json) |
| `--tokenizer` | `gpt2` | HuggingFace tokenizer name |
| `--model-size` | `small` | tiny, small, medium, large, xlarge, tiny-moe, medium-moe, large-moe |
| `--optimizer` | `velvet` | `velvet` or `adamw` |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--seq-len` | `512` | Sequence length (tokens per sample) |
| `--save-every` | `500` | Checkpoint every N steps (0 = epoch only) |
| `--output-dir` | `checkpoints` | Output directory |
| `--max-steps` | `0` | Max steps (0 = unlimited) |
| `--resume` | | Resume from checkpoint (.safetensors) |
| `--moe` | `false` | Enable Mixture of Experts |
| `--num-experts` | `8` | Number of experts (with --moe) |
| `--top-k` | `2` | Top-K experts per token (with --moe) |
| `--streaming` | `false` | Streaming mode for large files |
| `--chunk-mb` | `50` | Chunk size in MB (streaming mode) |
| `--cache` | | Path to pre-built binary cache |

### `vesper benchmark`

Runs Velvet vs AdamW head-to-head comparison. Outputs JSON report with per-step metrics.

### `vesper generate`

Autoregressive text generation from a trained checkpoint.

### `vesper cache build`

Pre-tokenizes a dataset into a memory-mapped binary cache for instant loading.

---

## Velvet Optimizer

The Velvet optimizer extends AdamW with three adaptive mechanisms:

1. **Entropy-Adaptive Learning Rate** — Scales LR based on the entropy of output logits. High entropy (model is uncertain) = higher LR to learn faster. Low entropy (model is confident) = lower LR for stability.

2. **Perplexity-Guided Momentum** — Adjusts beta1 based on current perplexity relative to a baseline. High perplexity = more aggressive momentum. Low perplexity = more conservative.

3. **Sparse-Aware Updates** — Optimized for FlyLoRA's sparse structure.

4. **Custom CUDA Kernels** — Fused parameter update kernel (momentum + weight decay + update in one pass), zero-copy GPU memory access.

5. **GPU-Optimized Gradient Clipping** — Accumulates gradient norms on GPU with a single CPU sync point.

### Benchmark Results (Velvet vs AdamW)

Local benchmark on RTX 4080 Laptop (12GB VRAM):

| Metric | AdamW | Velvet | Improvement |
|--------|-------|--------|-------------|
| Final Loss | 5.45 | **4.48** | **-17.7%** |
| Final Perplexity | 232 | **89** | **-62%** |
| Training Time | ~18s | ~19s | Similar |

---

## Mixture of Experts (MoE)

Each transformer layer's FFN is replaced by N expert FFNs and a learned router:

1. **Router**: Linear(hidden_size -> num_experts) + softmax
2. **Top-K Selection**: Select K experts per token based on router probabilities
3. **Weighted Sum**: Expert outputs weighted by normalized router probabilities
4. **Load Balancing Loss**: Switch Transformer auxiliary loss (`N * sum(f_i * P_i)`) prevents expert collapse

Each expert is a full GLU-style FFN with FlyLoRA + ERA activation, identical to the standard FFN.

```
Input → Router → Top-K indices + weights
      → Expert_0(input) × weight_0
      + Expert_1(input) × weight_1
      + ...
      → Output
```

When `--moe` is disabled (default), behavior is identical to a standard transformer.

### Optimized Token-Dispatch

The MoE routing uses **selective token-dispatch**: each expert only processes the tokens actually routed to it (via `index_select`), and results are scattered back with `scatter_add`. This avoids the naive approach of running every expert on all tokens then masking.

| Metric | Dense (tiny) | MoE naive | MoE optimized |
|--------|-------------|-----------|---------------|
| Time/step | ~0.14s | ~0.22s (+57%) | ~0.14s (~0%) |
| Speedup | — | — | **1.6x vs naive** |

With top-k=2 and N=4 experts, each expert processes ~50% of tokens instead of 100%, halving expert compute. MoE overhead is near-zero on GPU at this scale.

---

## Checkpointing and Resume

- **Periodic checkpoints**: `--save-every 500` saves every 500 steps
- **Epoch checkpoints**: Saved at every epoch end
- **Emergency checkpoints**: Auto-saved on SIGTERM/SIGINT (Vast.ai preemption)
- **Resume**: `--resume checkpoints/checkpoint-500.safetensors` continues from step 500
- **Training log**: `training_log.json` with per-step loss, loadable for plotting

```bash
# Resume from a checkpoint
./target/release/vesper train \
  --dataset fineweb-10M.txt \
  --tokenizer gpt2 \
  --model-size small \
  --resume checkpoints/checkpoint-500.safetensors \
  --epochs 3
```

---

## Cloud Training (Vast.ai)

### Using the setup script

```bash
# On a fresh Vast.ai A100 instance (nvidia/cuda:12.4.1-devel-ubuntu22.04)
bash scripts/vastai_setup.sh [HF_TOKEN]
```

This installs Rust, clones the repo, builds with `CUDA_ARCH=sm_80`, and downloads FineWeb-Edu.

### Using Docker

```bash
docker build -t vesper .
docker run --gpus all -v /data:/workspace vesper train \
  --dataset /workspace/fineweb-1B.txt \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --model-size large-moe \
  --epochs 3 --batch-size 8 --lr 3e-4 \
  --output-dir /workspace/checkpoints
```

---

## Plotting

```bash
# Training curves (loss + perplexity)
python scripts/plot_training.py checkpoints/training_log.json

# Benchmark comparison (4 graphs: loss, ppl, adaptive scales, grad norms)
python scripts/plot_benchmark.py benchmark_report.json

# Compare multiple runs
python scripts/plot_training.py run1/training_log.json run2/training_log.json
```

Requires: `pip install matplotlib numpy`

---

## Tech Stack

| Component | Version | Role |
|-----------|---------|------|
| **Python** | 3.10+ | Primary training language |
| **PyTorch** | 2.10+ | ML framework (tensors, autograd, CUDA) |
| **Triton** | 3.6+ | GPU kernel compiler (fused kernels) |
| **Flash Attention** | 2.8.3+ | Memory-efficient attention |
| Rust | 1.83+ | Alternative backend, inference |
| Candle | 0.9.1 (EricLBuehler fork) | Rust ML framework (small models, CPU) |
| tch-rs | 0.16 | Rust PyTorch bindings (GPU training) |
| CUDA Toolkit | 12.8 | GPU compute |
| Clap | 4.x | Rust CLI argument parsing |
| Wandb | | Experiment tracking (optional) |
| Tauri | 2.0 | Desktop app (optional) |

---

## Memory & Backend Comparison

| Model | Batch | Candle VRAM | PyTorch VRAM | Notes |
|-------|-------|-------------|--------------|-------|
| small (35M) | 4 | ~2GB | ~1GB | Both work fine |
| medium (89M) | 8 | ~6GB | ~3GB | Candle OK on RTX 4080 |
| large (350M) | 8 | ~25GB | ~8GB | Candle needs A100 |
| xlarge (1B) | 10 | ~70GB | ~18GB | **Candle impractical** |
| xlarge (1B) | 32 | OOM | ~45GB | PyTorch with grad ckpt |

**Why PyTorch wins for large models**: Candle's tape-based autograd retains ALL intermediate tensors (attention scores, FFN activations, residuals) until `loss.backward()` completes. For a 24-layer 1B model, this means 24 layers worth of activations stay in VRAM simultaneously. PyTorch frees each layer's activations as soon as its gradients are computed during backward, keeping only ~1 layer's worth of intermediates at any time.

## System Requirements

### Python Training (Recommended)
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (RTX 3060+)
- CUDA Toolkit 12.8
- `pip install -r python/requirements.txt`

### Rust CLI
- Rust 1.83+
- CUDA Toolkit 12.8
- For tch-rs backend: `pip install torch && export LIBTORCH_USE_PYTORCH=1`
- Visual Studio Build Tools 2022 (Windows) or GCC (Linux)

### Cloud (large models)
- NVIDIA H100 80GB or A100 80GB
- CUDA 12.4+
- 64GB+ RAM

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `nvcc not found` | Add CUDA bin to PATH: `export PATH=$PATH:/usr/local/cuda/bin` |
| `cl.exe not found` (Windows) | Open "x64 Native Tools Command Prompt for VS 2022" |
| `LNK1181: cannot open cuda.lib` | The `bindgen_cuda` patch in Cargo.toml fixes this |
| `LNK2038 RuntimeLibrary mismatch` | Use `--release` builds (debug has MT/MD conflicts) |
| `CUDA out of memory` | Reduce `--batch-size` or use smaller `--model-size` |
| `gather only supports contiguous tensors` | Fixed — `.contiguous()` added after narrow operations |

---

## Changelog

### v0.4.0 (February 2026)
- **Python training framework**: Complete PyTorch implementation with Flash Attn 2, Triton kernels
- **Triton kernels**: Fused Velvet update, fused cross-entropy (Unsloth-style), fused ERA activation
- **tch-rs backend**: Rust training via LibTorch/PyTorch autograd (feature-gated)
- Gradient accumulation (`--grad-accum`)
- Evaluation loop (`--val-dataset`, `--eval-every`)
- Wandb integration (`--wandb`)
- Multi-worker streaming dataset sharding

### v0.3.0 (February 2026)
- Mixture of Experts (MoE): ExpertFFN, MoELayer, top-K routing, load balancing loss
- MoE presets: tiny-moe, medium-moe, large-moe
- CLI flags: --moe, --num-experts, --top-k

### v0.2.0 (January-February 2026)
- vesper-cli: train, benchmark, generate, cache subcommands
- Velvet optimizer: backward_step, GPU gradient clipping, CUDA kernels
- ERA activation: GELU(x) + gamma * softplus(x)
- Streaming dataset loader for large files
- Binary dataset cache (memory-mapped)
- Checkpointing: periodic, epoch, emergency (SIGTERM)
- Resume from checkpoint with step continuity
- Training log (JSON) + plotting scripts
- Dockerfile for cloud deployment
- Vast.ai setup script
- FineWeb-Edu dataset downloader

### v0.1.0 (December 2025)
- Initial architecture: VesperLM, FlyLoRA, attention
- Tauri desktop application

---

**Built with Rust & Python | Powered by PyTorch & Candle | Accelerated by CUDA & Triton**

Made by [The Vesper House](https://github.com/thevesperhouse-hub)

## License

Proprietary - The Vesper House. All rights reserved. See [LICENSE](LICENSE).
