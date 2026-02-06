# VesperAI - Rust Edition

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.83+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Candle](https://img.shields.io/badge/Candle-0.9.1-blue.svg)](https://github.com/huggingface/candle)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-purple.svg)](https://tauri.app/)

> High-performance LLM training framework in pure Rust with Candle ML

---

## Project Status

### Functional
- **VesperLM** - Complete transformer model (Small/Medium/Large)
- **CUDA Training** - GPU training with autograd
- **Velvet Optimizer** - Enhanced AdamW with adaptive LR, custom CUDA kernels
- **FlyLoRA** - Sparse Low-Rank Adaptation (75% parameter reduction)
- **ERA Activation** - Entropy-Regularized Activation
- **CamemBERT Tokenizer** - French tokenization
- **Dataset Cache** - Memory-mapped binary cache (instant loading)
- **SafeTensors** - Model save/load
- **Chat Inference** - Text generation with top-p/top-k sampling
- **Tauri App** - Complete desktop interface

### Resolved Issues
- **MSVC Linker** - Correct Windows toolchain configuration
- **CUDA 12.8** - Compatibility via EricLBuehler/candle fork
- **bindgen_cuda** - Fix via guoqingbao/bindgen_cuda fork
- **Tensor Layout** - `.contiguous()` fixes after transpose operations
- **NaN Loss** - Numerical stability (epsilon, clamping, low LR)
- **Shape Mismatches** - FlyLoRA and ERA corrections

### Roadmap
- [ ] **French dataset** - Download Claire-Dialogue-French (gated, requires HuggingFace auth)
- [ ] **Generalization** - Train on more data to avoid overfitting
- [ ] **Multi-GPU** - NCCL support for distributed training
- [ ] **Quantization** - INT8/INT4 for faster inference
- [ ] **Streaming** - Token-by-token generation in chat

---

## Table of Contents

- [Project Status](#project-status)
- [Full Tech Stack](#full-tech-stack)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Architecture](#architecture)
- [Components](#components)
- [Configuration](#configuration)
- [Usage](#usage)
- [Benchmarks & Results](#benchmarks--results)
- [Troubleshooting](#troubleshooting)

---

## Full Tech Stack

### Backend (Rust)

| Component | Version | Role |
|-----------|---------|------|
| **Rust** | 1.83+ | Primary language, memory-safe, zero-cost abstractions |
| **Candle** | 0.9.1 (EricLBuehler fork) | Rust ML framework, GPU/CPU tensors |
| **cudarc** | 0.10 | Low-level CUDA bindings for Rust |
| **Tokio** | 1.x | Async runtime for non-blocking I/O |
| **Serde** | 1.x | JSON/binary serialization |
| **Tauri** | 2.0 | Desktop app framework (Rust backend) |

### Frontend (Web/Desktop)

| Component | Version | Role |
|-----------|---------|------|
| **React** | 18.2 | UI components |
| **TypeScript** | 5.3 | Frontend type safety |
| **Vite** | 5.0 | Ultra-fast build tool |
| **TailwindCSS** | 3.3 | Utility-first styling |
| **Lucide React** | 0.300 | Icons |
| **OGL** | 1.0 | WebGL for Aurora effects |

### GPU/CUDA

| Component | Version | Role |
|-----------|---------|------|
| **CUDA Toolkit** | 12.8 | Runtime and nvcc compiler |
| **cuDNN** | 9.x | Deep learning optimizations |
| **NCCL** | 2.x | Multi-GPU communication (optional) |

### Build Tools (Windows)

| Component | Version | Role |
|-----------|---------|------|
| **Visual Studio Build Tools** | 2022 (17.x) | MSVC compiler |
| **MSVC** | v143 | C++ toolchain |
| **Windows SDK** | 10.0.22621+ | System headers |
| **CMake** | 3.28+ | Build system (for dependencies) |

---

## System Requirements

### Windows 11/10

#### 1. Visual Studio Build Tools 2022

**Download**: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Required components** (check during installation):
```
Desktop development with C++
  |-- MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
  |-- Windows 11 SDK (10.0.22621.0) or Windows 10 SDK
  |-- C++ CMake tools for Windows
  |-- C++ ATL for latest v143 build tools (x86 & x64)
```

**Environment variables** (automatic after install):
```powershell
# Verify these paths exist
$env:VCToolsInstallDir  # e.g. C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\
```

#### 2. CUDA Toolkit 12.8

**Download**: https://developer.nvidia.com/cuda-12-8-0-download-archive

**Installation**:
```powershell
# Default path
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\

# Verify installation
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.8, V12.8.xxx
```

**Required environment variables**:
```powershell
# In system variables
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

# Add to PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
```

#### 3. Rust Toolchain

```powershell
# Install rustup (if not already installed)
winget install Rustlang.Rustup

# Install stable toolchain
rustup default stable

# Verify
rustc --version
# rustc 1.83.0 (90b35a623 2024-11-26)

cargo --version
# cargo 1.83.0 (5ffbef321 2024-10-29)
```

#### 4. Node.js (for the Tauri frontend)

```powershell
# Via winget
winget install OpenJS.NodeJS.LTS

# Verify
node --version
# v20.x.x

npm --version
# 10.x.x
```

---

## Installation

### Step 1: Clone the repo

```powershell
git clone https://github.com/thevesperhouse-hub/VelvetOptimizer.git
cd VelvetOptimizer
```

### Step 2: Verify prerequisites

```powershell
# Verification script
.\scripts\check-prereqs.ps1

# Or manually:
rustc --version          # >= 1.83
nvcc --version           # CUDA 12.8
cl.exe                   # MSVC available (open "x64 Native Tools Command Prompt")
node --version           # >= 20
```

### Step 3: Build Rust backend

```powershell
# Debug build (faster, for development)
cargo build

# Release build (optimized, for production)
cargo build --release

# Estimated build time:
# - Debug: ~3-5 min (first time)
# - Release: ~8-15 min (first time)
# - Incremental: ~10-30s
```

### Step 4: Setup frontend

```powershell
cd crates/vesper-app
npm install
```

### Step 5: Launch the application

```powershell
# Development mode (hot-reload)
npm run tauri dev

# Production build
npm run tauri build
```

---

## Architecture

```
VelvetOptimizer/
|-- Cargo.toml                 # Workspace config
|-- crates/
|   |-- vesper-core/           # VesperLM model
|   |   |-- src/
|   |   |   |-- model.rs       # Transformer architecture
|   |   |   |-- attention.rs   # Multi-head attention + RoPE
|   |   |   |-- flylora.rs     # Sparse LoRA (75% param reduction)
|   |   |   |-- era.rs         # Entropy-Regulated Activation
|   |   |-- Cargo.toml
|   |
|   |-- vesper-optimizer/      # Velvet Optimizer
|   |   |-- src/
|   |   |   |-- velvet.rs      # Enhanced AdamW with adaptive features
|   |   |   |-- cuda/          # Custom CUDA kernels
|   |   |       |-- mod.rs     # Rust wrapper
|   |   |       |-- kernels.cu # CUDA C++ code
|   |   |-- build.rs           # CUDA compilation script
|   |   |-- Cargo.toml
|   |
|   |-- vesper-metacog/        # Metacognition module
|   |   |-- src/
|   |       |-- meta_head.rs   # Error prediction head
|   |       |-- regulator.rs   # Adaptive regulator
|   |
|   |-- vesper-training/       # Training pipeline
|   |   |-- src/
|   |       |-- trainer.rs     # Training loop
|   |       |-- dataset.rs     # JSONL/JSON loading
|   |       |-- auto_scale.rs  # Chinchilla scaling laws
|   |
|   |-- vesper-app/            # Tauri application
|       |-- src/
|       |   |-- main.rs        # Tauri entry point
|       |   |-- commands.rs    # Rust <-> Frontend API
|       |   |-- App.tsx        # Main React UI
|       |   |-- components/    # React components
|       |-- package.json       # npm dependencies
|       |-- tauri.conf.json    # Tauri config
|       |-- tailwind.config.js # TailwindCSS config
|
|-- target/                    # Build output
    |-- debug/
    |-- release/
```

---

## Components

### 1. Candle ML Framework

**Why Candle over PyTorch?**
- **Performance**: No Python overhead, native Rust tensors
- **Memory safety**: No memory leaks, ownership system
- **Compilation**: AOT compilation, no JIT overhead
- **CUDA**: Native support via cudarc

**Fork used**: `EricLBuehler/candle` (rev 175926c9)
- Fixes for CUDA 12.8
- Better Windows support
- Optimizations for mistral.rs

```toml
# Cargo.toml
candle-core = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
candle-nn = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
```

**Critical patch for bindgen_cuda** (resolves CUDA linking errors on Windows):
```toml
# In Cargo.toml
[patch.crates-io]
bindgen_cuda = { git = "https://github.com/guoqingbao/bindgen_cuda.git" }
```

### 2. Velvet Optimizer

Custom optimizer based on AdamW with:
- **Entropy-adaptive LR**: Adjusts learning rate based on loss entropy
- **Perplexity-guided momentum**: Adaptive momentum based on perplexity
- **Sparse-aware updates**: Optimizations for FlyLoRA
- **Custom CUDA kernels**: Zero-copy GPU memory access, in-place updates

```rust
// vesper-optimizer/src/velvet.rs
pub struct VelvetOptimizer {
    params: Vec<Tensor>,
    m: HashMap<String, Tensor>,      // First moment
    v: HashMap<String, Tensor>,      // Second moment
    config: VelvetConfig,
    step: usize,
}
```

### 3. FlyLoRA (Sparse Low-Rank Adaptation)

75% parameter reduction via:
- Low-rank decomposition (A x B instead of W)
- Learned sparsity mask
- Adaptive rank per layer

```rust
// vesper-core/src/flylora.rs
pub struct FlyLoRALayer {
    base_weight: Tensor,      // Frozen weights
    lora_a: Tensor,           // Down projection (d x r)
    lora_b: Tensor,           // Up projection (r x d)
    sparsity_mask: Tensor,    // Binary mask
    rank: usize,              // LoRA rank (8-64)
}
```

### 4. ERA Activation (Entropy-Regularized Activation)

Alternative to GELU/SiLU with entropy regularization:

```rust
// vesper-core/src/era.rs
pub fn era_activation(x: &Tensor, temperature: f32) -> Result<Tensor> {
    // ERA = x * sigmoid(x/T) * (1 + entropy_term)
    let scaled = (x / temperature as f64)?;
    let gate = candle_nn::ops::sigmoid(&scaled)?;
    let base = (x * &gate)?;

    // Entropy term for regularization
    let entropy = compute_entropy(&gate)?;
    let regulated = (&base * (1.0 + entropy.to_scalar::<f32>()? * 0.1) as f64)?;

    Ok(regulated)
}
```

### 5. Tauri Desktop App

Modern frontend stack:
- **Tauri 2.0**: Secure, small footprint (~10MB), native
- **React 18**: Declarative UI
- **TailwindCSS**: Rapid styling
- **IPC**: Rust <-> JS communication via `invoke()`

```typescript
// Frontend -> Backend
const result = await invoke<BenchmarkResult>('start_benchmark', {
  config: { epochs: 10, model_size: 'Medium' }
});

// Backend events -> Frontend
await listen('benchmark-progress', (event) => {
  console.log(event.payload);
});
```

---

## Configuration

### Environment Variables

```powershell
# Required for CUDA
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME = $env:CUDA_PATH

# Optional: specific GPU architecture
$env:CUDA_ARCH = "sm_89"  # RTX 4090/4080
# $env:CUDA_ARCH = "sm_86"  # RTX 3090/3080
# $env:CUDA_ARCH = "sm_75"  # RTX 2080/2070
```

### Supported GPU Architectures

| GPU | Architecture | Code |
|-----|--------------|------|
| RTX 4090/4080/4070 | Ada Lovelace | sm_89 |
| RTX 3090/3080/3070 | Ampere | sm_86 |
| RTX 2080/2070/2060 | Turing | sm_75 |
| GTX 1080/1070 | Pascal | sm_61 |

### Cargo.toml Workspace

```toml
[workspace]
resolver = "2"
members = [
    "crates/vesper-core",
    "crates/vesper-optimizer",
    "crates/vesper-metacog",
    "crates/vesper-training",
    "crates/vesper-app",
]

[workspace.dependencies]
# Candle with CUDA
candle-core = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
candle-nn = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }

# CUDA bindings
cudarc = "0.10"

# Async
tokio = { version = "1", features = ["full"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Tauri
tauri = { version = "2.0", features = [] }

[profile.release]
opt-level = 3
lto = "fat"        # Link-time optimization
codegen-units = 1  # Better optimization
strip = true       # Reduce binary size

[patch.crates-io]
# Fix for bindgen CUDA
bindgen_cuda = { git = "https://github.com/guoqingbao/bindgen_cuda.git" }
```

---

## Usage

### Launch the application

```powershell
cd crates/vesper-app
npm run tauri dev
```

### AdamW vs Velvet Benchmark

The benchmark uses **real training with autograd** via `candle-nn`:
- **VarMap** + **VarBuilder** for parameters with gradient tracking
- **AdamW optimizer** from candle-nn with `backward_step()`
- **Real cross-entropy loss** on tokens
- **Perplexity** = exp(loss) displayed in real-time

**Velvet vs AdamW differences**:
| Parameter | AdamW | Velvet |
|-----------|-------|--------|
| Learning Rate | 1x | 1.5x (adaptive) |
| Beta1 (momentum) | 0.9 | 0.95 |
| Weight Decay | 0.01 | 0.01 |

**Usage**:
1. Load a dataset (JSON/JSONL SQuAD format supported)
2. Select epoch count
3. Click "AdamW vs Velvet"
4. Watch real-time logs with loss and perplexity

### Supported Dataset Formats

```json
// SQuAD format (recommended)
{
  "data": [
    {
      "paragraphs": [
        {
          "context": "The context text...",
          "qas": [
            {
              "question": "What is the question?",
              "answers": [{"text": "The answer"}]
            }
          ]
        }
      ]
    }
  ]
}

// Simple JSONL format
{"text": "First text example"}
{"text": "Second example"}
```

---

## Benchmarks & Results

### Test Configuration
- **GPU**: NVIDIA RTX 4080 (87% GPU utilization achieved)
- **CPU**: Intel i9-13900K
- **RAM**: 64GB DDR5
- **Dataset**: TinyStories (~37k tokens)

### VesperLM Model Sizes

| Size | Layers | Heads | Hidden | Params |
|------|--------|-------|--------|--------|
| Small | 6 | 6 | 384 | ~25M |
| **Medium** | 12 | 12 | 768 | **~89M** |
| Large | 24 | 16 | 1024 | ~350M |

### Training Results (120 epochs, Medium)

```
Epoch   1: loss=11.29 | ppl=79715
Epoch  30: loss=4.27  | ppl=71
Epoch  60: loss=2.37  | ppl=10.67
Epoch  90: loss=1.62  | ppl=5.04
Epoch 120: loss=1.22  | ppl=3.38
```

- **Total time**: ~2.5 minutes
- **Saved model**: 656 MB (SafeTensors)
- **GPU utilization**: 87% (optimal)

### Velvet vs AdamW Comparison

**Real benchmark (15 epochs, VesperLM Medium 89M params):**

| Metric | AdamW | Velvet | Improvement |
|--------|-------|--------|-------------|
| Final Loss | 6.38 | **5.39** | **-15.6%** |
| Final Perplexity | 591 | **219** | **-63%** |
| Time | 18.5s | 18.9s | Similar |
| Memory | 2000 MB | 2000 MB | Identical |

**Extended benchmark (20 epochs):**

| Metric | AdamW | Velvet | Improvement |
|--------|-------|--------|-------------|
| Final Loss | 5.45 | **4.48** | **-17.7%** |
| Final Perplexity | 232 | **89** | **-62%** |

**Velvet success factors:**
- Custom CUDA kernels (zero-copy, in-place updates)
- Adaptive learning rate (1.5x with entropy-guided adjustment)
- Adaptive momentum (beta1=0.95, perplexity-guided)
- Sparse-aware updates (optimized for FlyLoRA)

### Note on Overfitting

With a small dataset (37k tokens), the model reaches very low perplexity (3.38) but **overfits**. For better generalization:
- Use the **Claire-Dialogue-French** dataset (150M words)
- Or other large-scale corpora

---

## Troubleshooting

### Error: "nvcc not found"

```powershell
# Check CUDA_PATH
echo $env:CUDA_PATH
# Should show: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

# Add to PATH if missing
$env:PATH += ";$env:CUDA_PATH\bin"
```

### Error: "cl.exe not found" / MSVC Linker

```powershell
# Open "x64 Native Tools Command Prompt for VS 2022"
# Or load the environment manually:
& "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Verify MSVC is installed:
# Visual Studio Installer > Modify > "Desktop development with C++"
# Required components:
#   - MSVC v143 - VS 2022 C++ x64/x86 build tools
#   - Windows 11 SDK (10.0.22621.0)
#   - C++ CMake tools for Windows
```

### Error: "LINK : fatal error LNK1181: cannot open input file 'cuda.lib'"

```powershell
# The guoqingbao/bindgen_cuda fork resolves this issue
# Verify in Cargo.toml:
[patch.crates-io]
bindgen_cuda = { git = "https://github.com/guoqingbao/bindgen_cuda.git" }
```

### Error: "CUDA out of memory"

```powershell
# Reduce batch_size in the UI (max 8 recommended)
# Reduce seq_length (64 for benchmarks)
# Use a smaller model (Small instead of Large)
```

### Error: "NaN loss during training"

Fixes have been applied in the code:
- Epsilon added in logarithmic computations
- Value clamping to prevent overflow
- Reduced learning rate (max 0.0001)
- `.contiguous()` after transpose operations

### Error: "Tensor 'embedding' not found" (Chat)

VesperLM uses different tensor names:
- `embeddings.weight` (not `embedding`)
- `lm_head.weight` (not `output_proj`)

This fix has been applied in `commands.rs`.

### Error: "tokio runtime panic"

```rust
// Do not use reqwest::blocking in an async context
// Use reqwest async or std::thread::spawn
```

### Slow build

```powershell
# Use sccache for compilation caching
cargo install sccache
$env:RUSTC_WRAPPER = "sccache"

# Or use incremental builds
cargo build  # First build is slow
cargo build  # Subsequent builds are fast
```

### CamemBERT tokenizer not found

```powershell
# Download the CamemBERT tokenizer
huggingface-cli download camembert-base tokenizer.json

# Or manually copy to:
# C:\Users\<user>\AppData\Local\VesperAI\tokenizers\tokenizer.json
```

---

## Recommended Datasets

### For French (with CamemBERT tokenizer)

| Dataset | Size | Access | Usage |
|---------|------|--------|-------|
| **Claire-Dialogue-French** | 150M words | Gated (HuggingFace auth) | Conversational dialogue |
| SQuAD-FR | ~100k Q&A | Public | Question-Answering |
| French Wikipedia | ~2B words | Public | General text |

### Download Claire-Dialogue-French

```python
# 1. Accept conditions on HuggingFace:
# https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1

# 2. Login and download:
from datasets import load_dataset
from huggingface_hub import login

login(token="hf_XXXXX")  # Token from huggingface.co/settings/tokens
ds = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1")

# 3. Export to TXT
with open("claire_train.txt", "w", encoding="utf-8") as f:
    for example in ds["train"]:
        f.write(example["text"] + "\n")
```

---

## Credits

- **Hugging Face Candle** - Rust ML framework
- **EricLBuehler** - Candle fork with CUDA 12.8 fixes
- **Tauri** - Desktop app framework
- **guoqingbao** - bindgen_cuda fix for Windows
- **OpenLLM-France** - Claire-Dialogue-French dataset

---

## Changelog

### v0.2.0 (January 2026)
- VesperLM complete with attention, FlyLoRA, ERA
- CUDA training functional with autograd
- CamemBERT tokenizer integrated
- Chat inference with top-p/top-k sampling
- Memory-mapped dataset cache
- Console logs without limit + auto-scroll
- Fix MSVC linker / bindgen_cuda
- Fix NaN loss (numerical stability)
- Fix tensor shapes (FlyLoRA, ERA)

### v0.1.0 (December 2025)
- Initial release
- Base architecture

---

**Built with Rust | Powered by Candle | Accelerated by CUDA**

Made by [The Vesper House](https://github.com/thevesperhouse-hub)

## License

Proprietary - The Vesper House. All rights reserved. See [LICENSE](LICENSE).
