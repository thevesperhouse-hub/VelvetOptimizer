# VelvetOptimizer

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.83+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Candle](https://img.shields.io/badge/Candle-0.9.1-blue.svg)](https://github.com/huggingface/candle)

> High-performance LLM training framework in pure Rust with a custom optimizer, Mixture of Experts, and CUDA kernels.

VelvetOptimizer is a from-scratch transformer training system built around the **Velvet optimizer** — an enhanced AdamW with entropy-adaptive learning rate, perplexity-guided momentum, and custom CUDA kernels. It includes **VesperLM**, a transformer architecture with FlyLoRA (sparse low-rank adaptation), ERA activation, and Mixture of Experts (MoE).

---

## Features

- **Velvet Optimizer** — AdamW + entropy-adaptive LR + perplexity-guided momentum + sparse awareness + custom CUDA kernels
- **VesperLM** — Transformer with multi-head attention, RoPE, FlyLoRA, ERA activation
- **Mixture of Experts (MoE)** — Top-K routing, N expert FFNs per layer, Switch Transformer load balancing loss
- **CLI Training** — `vesper train` with checkpointing, resume, SIGTERM auto-save
- **Streaming** — Chunk-by-chunk training for large datasets (>1GB)
- **Binary Cache** — Memory-mapped dataset cache for instant loading
- **Benchmarking** — `vesper benchmark` for Velvet vs AdamW comparison
- **Text Generation** — `vesper generate` with temperature/top-p sampling
- **Tauri Desktop App** — GUI for training and inference
- **Cloud Ready** — Dockerfile + Vast.ai setup script for A100 training

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
├── crates/
│   ├── vesper-core/           # Model architecture
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
│   │   ├── train.rs           # Training pipeline (resume, SIGTERM, checkpointing)
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
│   └── vastai_setup.sh        # One-command Vast.ai A100 setup
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
| Rust | 1.83+ | Primary language |
| Candle | 0.9.1 (EricLBuehler fork) | ML framework (tensors, autograd, CUDA) |
| CUDA Toolkit | 12.8 | GPU compute |
| Clap | 4.x | CLI argument parsing |
| SafeTensors | | Model checkpoint format |
| Tauri | 2.0 | Desktop app (optional) |
| React + TypeScript | 18 / 5.3 | Frontend UI (optional) |

---

## System Requirements

### Minimum (CPU training)
- Rust 1.83+
- 8GB RAM

### Recommended (GPU training)
- NVIDIA GPU with 8GB+ VRAM (RTX 3060+)
- CUDA Toolkit 12.8
- Visual Studio Build Tools 2022 (Windows) or GCC (Linux)
- 16GB+ RAM

### Cloud (large models)
- NVIDIA A100 40/80GB
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

**Built with Rust | Powered by Candle | Accelerated by CUDA**

Made by [The Vesper House](https://github.com/thevesperhouse-hub)

## License

Proprietary - The Vesper House. All rights reserved. See [LICENSE](LICENSE).
