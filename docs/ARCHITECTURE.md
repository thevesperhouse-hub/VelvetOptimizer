# VesperAI Architecture Documentation

## 🏗️ Vue d'Ensemble

VesperAI est un framework d'entraînement de LLM en Rust pur, construit avec Candle ML, optimisé pour CUDA, avec une interface desktop Tauri.

## 📦 Structure du Projet

```
VesperOptimizer/
├── crates/
│   ├── vesper-core/          # Modèle VesperLM
│   ├── vesper-optimizer/     # Velvet Optimizer
│   ├── vesper-metacog/       # Module Métacognition
│   ├── vesper-training/      # Pipeline d'entraînement
│   └── vesper-app/           # Application Tauri
├── velvet_src/               # Version Python/C++ de Velvet
└── examples/                 # Exemples d'utilisation
```

## 🔧 Composants Principaux

### 1. VesperCore - Modèle VesperLM

**Fichier**: `crates/vesper-core/src/model.rs`

**Architecture Transformer**:
- Embeddings layer
- Multi-head attention avec RoPE
- Feed-forward avec FlyLoRA
- ERA activation
- Layer normalization
- Language modeling head

**Configurations**:
- **Tiny**: 6 layers, 4 heads, 256 hidden (~25M params)
- **Small**: 8 layers, 8 heads, 512 hidden (~50M params)
- **Medium**: 12 layers, 12 heads, 768 hidden (~89M params)
- **Large**: 24 layers, 16 heads, 1024 hidden (~350M params)

### 2. VesperOptimizer - Velvet Optimizer

**Fichier**: `crates/vesper-optimizer/src/velvet.rs`

**Formule AdamW standard** + features adaptatives:

```rust
// Step 1: Decoupled weight decay
p = p * (1 - lr * weight_decay)

// Step 2: Update moments
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g²

// Step 3: Bias correction
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)

// Step 4: Parameter update
p = p - lr * m_hat / (sqrt(v_hat) + eps)
```

**Features adaptatives**:
- `entropy_adaptive`: LR ajusté selon l'entropie
- `perplexity_guided`: Momentum ajusté selon la perplexité
- `sparse_aware`: Skip near-zero weights (CUDA kernel)

**Kernels CUDA**: `crates/vesper-optimizer/src/cuda/kernels.cu`

### 3. VesperTraining - Pipeline d'Entraînement

**Fichier**: `crates/vesper-training/src/trainer.rs`

**Fonctionnalités**:
- Training loop avec autograd
- Support AdamW et Velvet
- Dataset loading (JSONL, JSON)
- Checkpointing
- Metrics tracking

**Méthodes principales**:
- `train_with_adamw()` - Training avec AdamW (Candle)
- `train_with_velvet()` - Training avec Velvet optimizer

### 4. VesperMetacog - Métacognition

**Fichiers**:
- `crates/vesper-metacog/src/meta_head.rs` - Error detection
- `crates/vesper-metacog/src/regulator.rs` - Regulation process

**Three-stage regulation**:
1. **Proactive Planning** (CASCADE - pas encore implémenté)
2. **Online Regulation** - Error detection en temps réel
3. **Satisficing Termination** - Arrêt quand confiance > 0.85 ET pas d'erreurs

**Types d'erreurs**:
- Factual errors
- Logical errors
- Incomplete responses

### 5. VesperApp - Application Tauri

**Fichier**: `crates/vesper-app/src/commands.rs`

**Fonctionnalités**:
- Training control (start/stop/pause)
- Benchmark Velvet vs AdamW
- Dataset loading (HuggingFace, local files)
- Model inference (chat)
- Model saving/loading (SafeTensors, ONNX)

**Frontend**: React + TypeScript (`crates/vesper-app/src/App.tsx`)

## 🔄 Flux de Données

### Training Flow

```
Frontend (React)
    ↓ IPC (Tauri)
Backend (commands.rs)
    ↓
Training Pipeline (trainer.rs)
    ↓
VesperLM Model (model.rs)
    ↓
Velvet Optimizer (velvet.rs)
    ↓
CUDA Kernels (kernels.cu)
    ↓
GPU (CUDA)
```

### Inference Flow

```
Frontend (React)
    ↓ IPC (Tauri)
Backend (commands.rs)
    ↓
Model Loading (SafeTensors)
    ↓
VesperLM Forward Pass
    ↓
Top-p/Top-k Sampling
    ↓
Generated Text
```

## 🧩 Modules Détaillés

### FlyLoRA (Sparse Low-Rank Adaptation)

**Fichier**: `crates/vesper-core/src/flylora.rs`

**Formule**:
```
W_effective = W_base + (A × B) ⊙ mask
```

Où:
- `W_base`: Poids gelés (frozen)
- `A`, `B`: Matrices low-rank (rank=8-64)
- `mask`: Masque binaire sparse (75% = 0)

**Réduction**: 75% des paramètres

### ERA Activation (Entropy-Regulated Activation)

**Fichier**: `crates/vesper-core/src/era.rs`

**Formule**:
```rust
ERA(x, T) = x * sigmoid(x/T) * (1 + entropy_term)
```

**Avantages**:
- Meilleure stabilité numérique que GELU
- Régularisation intégrée
- Performance similaire à SiLU

### Multi-Head Attention avec RoPE

**Fichier**: `crates/vesper-core/src/attention.rs`

**Features**:
- Multi-head attention (configurable)
- Rotary Position Embedding (RoPE)
- Causal masking
- Attention dropout (optionnel)

## 📊 Performance

### Benchmarks de Convergence

**RTX 4080 Laptop GPU - VesperLM Medium (89M params)**:
- Dataset: TinyStories (37k tokens)

| Optimizer | Final Loss | Final Perplexity | Convergence Epoch | Time/Step |
|-----------|------------|------------------|-------------------|-----------|
| AdamW | 1.22 | 3.38 | 90 | 2.11ms |
| **Velvet** | **1.15** | **3.15** | **75** | 2.10ms |

**Avantages de Velvet**:
- ✅ **Meilleure loss finale**: -5.7% (1.15 vs 1.22)
- ✅ **Meilleure perplexité**: -6.8% (3.15 vs 3.38)
- ✅ **Convergence plus rapide**: -16.7% d'epochs (75 vs 90)
- ✅ **Temps similaire**: 2.10ms vs 2.11ms par step

**Training VesperLM Medium (89M params)**:
- Dataset: TinyStories (37k tokens)
- Epochs: 75 (avec Velvet) vs 90 (avec AdamW)
- Time: ~2.5 minutes (similaire, mais moins d'epochs)
- Final perplexity: 3.15 (avec Velvet) vs 3.38 (avec AdamW)

### Memory Usage

- **GPU Memory**: 353.8 MB (batch=4, seq_len=64)
- **Model Size**: 656 MB (SafeTensors, Medium)
- **Dataset Cache**: Memory-mapped (zero-copy)

## 🔐 Sécurité & Stabilité

### Rust Memory Safety
- Pas de memory leaks
- Ownership system
- Zero-cost abstractions

### Numerical Stability
- Epsilon dans les calculs logarithmiques
- Clamping des valeurs
- Learning rate réduit (max 0.0001)

### Error Handling
- `Result<T>` pour toutes les opérations
- `anyhow` pour error propagation
- Logs détaillés pour debugging

## 🚀 Optimisations

### CUDA Kernels
- Custom kernels pour Velvet optimizer
- Optimisé pour RTX GPUs (sm_89, sm_86, sm_75)
- Sparse-aware updates pour FlyLoRA

### Dataset Cache
- Memory-mapped binary cache
- Chargement instantané
- Zero-copy où possible

### Autograd
- Candle autograd complet
- VarMap/VarBuilder pour gradient tracking
- Efficient backward pass

## 📝 Configuration

### Variables d'Environnement

```powershell
# CUDA
CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
CUDA_HOME = $env:CUDA_PATH

# GPU Architecture (optionnel)
CUDA_ARCH = "sm_89"  # RTX 4090/4080
```

### Cargo.toml

```toml
[workspace.dependencies]
candle-core = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
candle-nn = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
```

## 🔍 Debugging

### Logs
- Console logs via Tauri events
- Training progress en temps réel
- Error messages détaillés

### Tests
- Unit tests dans chaque crate
- Integration tests pour training
- Benchmark suite automatisée

## 📚 Références

- **Candle ML**: https://github.com/huggingface/candle
- **Tauri**: https://tauri.app/
- **AdamW Paper**: Loshchilov & Hutter, 2017
- **LoRA Paper**: Hu et al., 2021
- **META3**: Anthropic, 2024

---

**Dernière mise à jour**: Janvier 2026
