<div align="center">

# Velvet Optimizer

### High-Performance LLM Training from Scratch

**Custom optimizer. Custom kernels. One framework.**

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-3.6-624FE8)](https://github.com/triton-lang/triton)
[![Rust](https://img.shields.io/badge/Rust-1.83+-DEA584?logo=rust&logoColor=white)](https://www.rust-lang.org/)

---

*VelvetOptimizer is a from-scratch LLM training framework built around the **Velvet optimizer** — an enhanced AdamW with entropy-adaptive learning rate, perplexity-guided momentum, and fused GPU kernels. It ships with **VesperLM**, a 1.3B-parameter transformer featuring FlyLoRA, ERA activation, Flash Attention 2, and Mixture of Experts.*

[Getting Started](#getting-started) | [Architecture](#architecture) | [Velvet Optimizer](#the-velvet-optimizer) | [Benchmarks](#benchmarks)

</div>

---

## Highlights

| | Feature | What it does |
|---|---------|-------------|
| **1** | **Velvet Optimizer** | AdamW + entropy-adaptive LR + perplexity-guided momentum. Fused Triton kernel — one GPU pass per parameter. |
| **2** | **FlyLoRA** | Sparse low-rank adaptation on every FFN projection. Full base + lightweight LoRA branches, trained jointly from scratch. |
| **3** | **ERA Activation** | `GELU(x) + gamma * softplus(x)` — entropy-regularized activation with fused Triton kernel. |
| **4** | **Flash Attention 2** | Via PyTorch SDPA. O(n) memory, no materializing the full attention matrix. |
| **5** | **Mixture of Experts** | Top-K routing, N expert FFNs, Switch Transformer load balancing. Scale params without scaling compute. |
| **6** | **Fused Cross-Entropy** | Unsloth-style chunked CE — saves ~17GB for 128K vocab. Falls back to PyTorch for smaller vocabs. |
| **7** | **Gradient Checkpointing** | Recompute activations during backward. Cuts activation memory by ~60%. |
| **8** | **Streaming + Caching** | Stream multi-GB datasets line-by-line with worker sharding, or tokenize-once with memory-mapped cache. |

---

## Getting Started

### Install

```bash
git clone https://github.com/thevesperhouse-hub/VelvetOptimizer.git
cd VelvetOptimizer/python
pip install -r requirements.txt   # PyTorch CUDA 12.8 + Triton 3.6
pip install -e .
```

### Train

```bash
# 1.3B model, bf16, streaming, wandb tracking
python train.py \
    --dataset data.txt \
    --tokenizer gpt2 \
    --model-size xlarge \
    --dtype bf16 \
    --gradient-checkpointing \
    --batch-size 8 \
    --seq-len 4096 \
    --lr 1e-4 \
    --max-steps 50000 \
    --wandb \
    --data-mode streaming
```

```bash
# MoE — 8 experts, top-2 routing
python train.py \
    --dataset data.txt \
    --model-size large \
    --moe --num-experts 8 --top-k 2 \
    --dtype bf16
```

```bash
# Resume from checkpoint
python train.py \
    --dataset data.txt \
    --model-size xlarge \
    --resume checkpoints/step_5000.pt
```

### What you get

```
Training:  12%|████████░░░░░░░░░░░░| 6000/50000 [2:14:30<18:40:00] loss=4.21, ppl=67, tok/s=18200, VRAM=62G
```

Live tqdm progress bar. Loss, perplexity, throughput, VRAM — updated every step. Detailed logs at `--log-every` intervals. Wandb dashboards with `--wandb`. Checkpoints auto-saved on SIGINT/SIGTERM.

---

## The Velvet Optimizer

Standard AdamW applies the same learning rate and momentum to every parameter regardless of how well the model is learning. Velvet adapts in real time:

### Entropy-Adaptive Learning Rate

The model's output entropy tells us how uncertain it is. Velvet scales the LR proportionally:

- **High entropy** (model is guessing) → LR scales up → faster exploration
- **Low entropy** (model is confident) → LR scales down → stable refinement

```
effective_lr = base_lr * clamp(entropy / target_entropy, 0.7, 1.3)
```

### Perplexity-Guided Momentum

Momentum (beta1) is adjusted based on the current perplexity:

- **High perplexity** (poor predictions) → more aggressive momentum → push through
- **Low perplexity** (good predictions) → conservative momentum → preserve gains

### Fused GPU Kernel

The entire optimizer step — moment updates, bias correction, weight decay, adaptive scaling — runs in a **single Triton kernel launch** per parameter. Three backends, auto-detected:

| Priority | Backend | When |
|----------|---------|------|
| 1st | **Triton** | CUDA GPU + Triton installed |
| 2nd | **CUDA native** | CUDA GPU, no Triton (compiled via `cpp_extension`) |
| 3rd | **PyTorch** | CPU or no GPU extensions |

### Benchmark: Velvet vs AdamW

Trained on FineWeb-Edu, same model, same hyperparameters, same hardware:

| Metric | AdamW | Velvet | Delta |
|--------|------:|-------:|------:|
| Final Loss | 5.45 | **4.48** | **-17.7%** |
| Perplexity | 232 | **89** | **-61.6%** |
| Wall Time | ~18s | ~19s | +5% |

> 62% lower perplexity for 5% more wall time. The adaptive mechanisms converge faster by investing LR where it matters.

---

## Architecture

### VesperLM

```
Input IDs → Embedding → [TransformerLayer × 24] → LayerNorm → LM Head → Logits
                              │
                              ├── Pre-Norm → Multi-Head Attention (Flash Attn 2 + RoPE)
                              │                   Q, K, V, O projections
                              │
                              └── Pre-Norm → FeedForward (or MoE)
                                                ├── gate_proj (FlyLoRA) → ERA
                                                ├── up_proj   (FlyLoRA)
                                                └── down_proj → output
```

### Model Sizes

| Config | Layers | Heads | Hidden | FFN | Params | Hardware |
|--------|-------:|------:|-------:|----:|-------:|----------|
| `tiny` | 6 | 4 | 256 | 1024 | ~30M | Any GPU |
| `small` | 8 | 8 | 512 | 2048 | ~100M | 8GB+ |
| `medium` | 12 | 12 | 768 | 3072 | ~300M | 12GB+ |
| `large` | 24 | 16 | 1024 | 4096 | ~700M | 24GB+ |
| `xlarge` | 24 | 16 | 2048 | 5504 | **~1.3B** | 48GB+ |

### FlyLoRA

Sparse low-rank adaptation applied to every FFN projection. Combines full base weights with lightweight LoRA branches for efficient joint training from scratch.

Each FFN projection carries a trainable low-rank adapter:

```
output = base_linear(x) + (alpha/rank) * lora_up(lora_down(x))
```

- `rank=16`, `alpha=32` by default
- Sparse random projection buffer (25% sparsity) for initialization
- Base weights + LoRA train jointly from scratch

### ERA Activation

**E**ntropy-**R**egularized **A**ctivation:

```
ERA(x) = GELU(x) + gamma * softplus(x)
```

The softplus tail prevents dead neurons and maintains gradient flow in deep networks. `gamma=0.1` by default. Fused into a single Triton kernel (forward + backward).

### Mixture of Experts

When `--moe` is enabled, each layer's FFN is replaced by N parallel expert FFNs + a learned router:

```
tokens → Router (softmax) → Top-K selection → Dispatch to K experts → Weighted sum → output
```

- **Selective dispatch**: each expert only processes its assigned tokens (not all tokens)
- **Load balancing**: Switch Transformer auxiliary loss prevents expert collapse
- Top-K=2 with N=8 experts: 8x parameters, 2x compute per token

---

## Training Features

### Data Loading

| Mode | Flag | Use Case |
|------|------|----------|
| **Streaming** | `--data-mode streaming` | Large files (>1GB). Line-by-line, never loads full file. |
| **In-Memory** | `--data-mode inmemory` | Small files. Full tokenization upfront. |
| **Cached** | `--data-mode cached` | Tokenize once, memory-map `.pt` file on subsequent runs. |
| **JSONL** | `--data-mode jsonl` | Each line is `{"text": "..."}`. Concatenated and chunked. |

Streaming supports multi-worker sharding (`--num-workers 4`) with auto-recovery on worker timeouts.

### Mixed Precision

| Flag | Type | Use Case |
|------|------|----------|
| `--dtype bf16` | BFloat16 | **Recommended.** No loss scaling needed. |
| `--dtype fp16` | Float16 | Auto GradScaler. Wider hardware support. |
| `--dtype fp32` | Float32 | Baseline. 2x memory. |

### Checkpointing & Resume

- `--save-every 5000` — periodic checkpoint saves
- Auto-save on `Ctrl+C` (SIGINT) and SIGTERM (cloud preemption)
- `--resume checkpoints/step_5000.pt` — resume training from any checkpoint
- Best model tracked automatically

### Experiment Tracking

```bash
python train.py ... --wandb --wandb-project my-project --wandb-run run-name
```

Logs: loss, perplexity, LR, effective LR, gradient norm, tokens/sec, VRAM usage.

---

## Project Structure

```
VelvetOptimizer/
├── python/                         # Python training framework
│   ├── train.py                    # CLI training script
│   ├── requirements.txt
│   ├── setup.py
│   └── vesper/
│       ├── model.py                # VesperLM (Flash Attn 2, RoPE, FlyLoRA, MoE)
│       ├── optimizer.py            # Velvet optimizer (torch.optim.Optimizer)
│       ├── config.py               # Model configs + presets
│       ├── data.py                 # Datasets (streaming, cached, JSONL)
│       └── kernels/
│           ├── velvet_triton.py    # Fused optimizer step (Triton)
│           ├── era_triton.py       # Fused ERA activation (Triton)
│           ├── fused_ce.py         # Chunked cross-entropy (Triton)
│           └── velvet_cuda.py      # CUDA native fallback
│
├── crates/                         # Rust implementation
│   ├── vesper-core/                # Model architecture (Candle)
│   ├── vesper-optimizer/           # Velvet optimizer + CUDA kernels
│   ├── vesper-training/            # Training infrastructure
│   ├── vesper-cli/                 # CLI (train, benchmark, generate)
│   ├── vesper-app/                 # Tauri desktop app
│   └── vesper-metacog/             # Metacognition module
│
├── scripts/                        # Utilities
│   ├── download_fineweb.py         # FineWeb-Edu dataset downloader
│   ├── plot_benchmark.py           # Benchmark visualization
│   ├── plot_training.py            # Training curves
│   └── vastai_setup.sh             # One-command cloud setup
│
└── Dockerfile                      # Multi-stage CUDA build
```

---

## Cloud Training

### Vast.ai / RunPod (one-liner)

```bash
bash scripts/vastai_setup.sh
```

### Manual setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install triton tokenizers tqdm wandb
git clone https://github.com/thevesperhouse-hub/VelvetOptimizer.git
cd VelvetOptimizer/python && pip install -e .
```

### Recommended configs by GPU

| GPU | VRAM | Model | Batch | Seq Len | tok/s |
|-----|-----:|-------|------:|--------:|------:|
| RTX 4080 | 12GB | medium | 8 | 512 | ~25K |
| RTX 4090 | 24GB | large | 8 | 2048 | ~20K |
| A100 80GB | 80GB | xlarge | 12 | 4096 | ~30K |
| H100 80GB | 80GB | xlarge | 16 | 4096 | ~50K |
| RTX PRO 6000 | 96GB | xlarge | 8 | 4096 | ~18K |

---

## Rust Backend

The Rust implementation includes two backends for environments where Python isn't ideal:

```bash
# Candle (pure Rust, no Python dependency)
cargo build --release -p vesper-cli
./target/release/vesper train --dataset data.txt --model-size small

# tch-rs (Rust + PyTorch autograd, GPU training)
pip install torch && export LIBTORCH_USE_PYTORCH=1
cargo build --release -p vesper-cli --features tch-backend
./target/release/vesper train --backend tch --dataset data.txt --model-size xlarge --dtype bf16
```

CLI commands: `train`, `benchmark`, `generate`, `cache build`, `cache info`.

---

## Changelog

### v0.5.0 — February 2026
- NaN-safe training: auto-skip divergent steps, tighter adaptive scaling [0.7, 1.3]
- Anti-deadlock DataLoader: 60s worker timeout + auto-restart
- `torch.compile()` support (auto-disabled with gradient checkpointing)
- tqdm live progress bar
- Fixed `total_params()` estimate (was 712M, actual 1.3B)

### v0.4.0 — February 2026
- Python training framework: complete PyTorch implementation
- Triton kernels: fused Velvet update, fused cross-entropy, fused ERA
- tch-rs backend: Rust training via LibTorch (feature-gated)
- Gradient accumulation, evaluation loop, wandb integration

### v0.3.0 — February 2026
- Mixture of Experts: top-K routing, load balancing loss, MoE presets

### v0.2.0 — January 2026
- vesper-cli: train, benchmark, generate, cache
- Streaming dataset, binary cache, checkpointing, resume
- Docker + Vast.ai deployment

### v0.1.0 — December 2025
- Initial architecture: VesperLM, FlyLoRA, ERA, attention

---

<div align="center">

*No, we didn't just wrap HuggingFace. Yes, we wrote the CUDA kernels. No, we don't sleep much.*

---

**Built by [The Vesper House](https://github.com/thevesperhouse-hub)** — powered by mass amounts of ☕ and mass amounts of `NaN`

*Proprietary License. All rights reserved.*

</div>
