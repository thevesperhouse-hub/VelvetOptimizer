# 🔥 VesperAI - Rust Edition

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.83+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Candle](https://img.shields.io/badge/Candle-0.9.1-blue.svg)](https://github.com/huggingface/candle)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-purple.svg)](https://tauri.app/)

> Framework d'entraînement LLM haute performance en Rust pur avec Candle ML

---

## 🎯 État du Projet

### ✅ Fonctionnel
- **VesperLM** - Modèle transformer complet (Small/Medium/Large)
- **Training CUDA** - Entraînement GPU avec autograd
- **Velvet Optimizer** - AdamW amélioré avec LR adaptatif
- **FlyLoRA** - Sparse Low-Rank Adaptation (75% réduction params)
- **ERA Activation** - Entropy-Regularized Activation
- **CamemBERT Tokenizer** - Tokenization français
- **Dataset Cache** - Memory-mapped binary cache (chargement instantané)
- **SafeTensors** - Sauvegarde/chargement modèles
- **Chat Inference** - Génération de texte avec top-p/top-k sampling
- **Application Tauri** - Interface desktop complète

### 🔧 Problèmes Résolus
- **MSVC Linker** - Configuration correcte des toolchains Windows
- **CUDA 12.8** - Compatibilité via fork EricLBuehler/candle
- **bindgen_cuda** - Fix via fork guoqingbao/bindgen_cuda
- **Tensor Layout** - Corrections `.contiguous()` après transpose
- **NaN Loss** - Stabilité numérique (epsilon, clamping, LR bas)
- **Shape Mismatches** - FlyLoRA et ERA corrigés

### ⏳ Reste à Faire
- [ ] **Dataset français** - Télécharger Claire-Dialogue-French (gated, nécessite auth HuggingFace)
- [ ] **Généralisation** - Entraîner sur plus de données pour éviter l'overfitting
- [ ] **Multi-GPU** - Support NCCL pour entraînement distribué
- [ ] **Quantization** - INT8/INT4 pour inférence plus rapide
- [ ] **Streaming** - Génération token par token dans le chat

---

## 📋 Table des Matières

- [État du Projet](#-état-du-projet)
- [Stack Technique Complète](#-stack-technique-complète)
- [Prérequis Système](#-prérequis-système)
- [Installation Détaillée](#-installation-détaillée)
- [Architecture](#-architecture)
- [Composants](#-composants)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Troubleshooting](#-troubleshooting)

---

## 🛠 Stack Technique Complète

### Backend (Rust)

| Composant | Version | Rôle |
|-----------|---------|------|
| **Rust** | 1.83+ | Langage principal, memory-safe, zero-cost abstractions |
| **Candle** | 0.9.1 (EricLBuehler fork) | Framework ML Rust, tenseurs GPU/CPU |
| **cudarc** | 0.10 | Bindings CUDA low-level pour Rust |
| **Tokio** | 1.x | Runtime async pour I/O non-bloquant |
| **Serde** | 1.x | Sérialisation JSON/binaire |
| **Tauri** | 2.0 | Framework desktop app (Rust backend) |

### Frontend (Web/Desktop)

| Composant | Version | Rôle |
|-----------|---------|------|
| **React** | 18.2 | UI components |
| **TypeScript** | 5.3 | Type safety frontend |
| **Vite** | 5.0 | Build tool ultra-rapide |
| **TailwindCSS** | 3.3 | Styling utility-first |
| **Lucide React** | 0.300 | Icônes |
| **OGL** | 1.0 | WebGL pour effets Aurora |

### GPU/CUDA

| Composant | Version | Rôle |
|-----------|---------|------|
| **CUDA Toolkit** | 12.8 | Runtime et compilateur nvcc |
| **cuDNN** | 9.x | Optimisations deep learning |
| **NCCL** | 2.x | Multi-GPU communication (optionnel) |

### Build Tools (Windows)

| Composant | Version | Rôle |
|-----------|---------|------|
| **Visual Studio Build Tools** | 2022 (17.x) | Compilateur MSVC |
| **MSVC** | v143 | Toolchain C++ |
| **Windows SDK** | 10.0.22621+ | Headers système |
| **CMake** | 3.28+ | Build system (pour dépendances) |

---

## 💻 Prérequis Système

### Windows 11/10

#### 1. Visual Studio Build Tools 2022

**Téléchargement**: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Composants requis** (cocher lors de l'installation):
```
☑ Desktop development with C++
  ├── MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
  ├── Windows 11 SDK (10.0.22621.0) ou Windows 10 SDK
  ├── C++ CMake tools for Windows
  └── C++ ATL for latest v143 build tools (x86 & x64)
```

**Variables d'environnement** (automatiques après install):
```powershell
# Vérifier que ces paths existent
$env:VCToolsInstallDir  # ex: C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\
```

#### 2. CUDA Toolkit 12.8

**Téléchargement**: https://developer.nvidia.com/cuda-12-8-0-download-archive

**Installation**:
```powershell
# Chemin par défaut
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\

# Vérifier l'installation
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.8, V12.8.xxx
```

**Variables d'environnement requises**:
```powershell
# Dans les variables système
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

# Ajouter au PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
```

#### 3. Rust Toolchain

```powershell
# Installer rustup (si pas déjà fait)
winget install Rustlang.Rustup

# Installer la toolchain stable
rustup default stable

# Vérifier
rustc --version
# rustc 1.83.0 (90b35a623 2024-11-26)

cargo --version
# cargo 1.83.0 (5ffbef321 2024-10-29)
```

#### 4. Node.js (pour le frontend Tauri)

```powershell
# Via winget
winget install OpenJS.NodeJS.LTS

# Vérifier
node --version
# v20.x.x

npm --version
# 10.x.x
```

---

## 📥 Installation Détaillée

### Étape 1: Cloner le repo

```powershell
git clone https://github.com/thevesperhouse-hub/VesperAI.git
cd VesperAI
```

### Étape 2: Vérifier les prérequis

```powershell
# Script de vérification
.\scripts\check-prereqs.ps1

# Ou manuellement:
rustc --version          # >= 1.83
nvcc --version           # CUDA 12.8
cl.exe                   # MSVC disponible (ouvrir "x64 Native Tools Command Prompt")
node --version           # >= 20
```

### Étape 3: Build backend Rust

```powershell
# Build debug (plus rapide, pour dev)
cargo build

# Build release (optimisé, pour prod)
cargo build --release

# Temps de build estimé:
# - Debug: ~3-5 min (première fois)
# - Release: ~8-15 min (première fois)
# - Incrémental: ~10-30s
```

### Étape 4: Setup frontend

```powershell
cd crates/vesper-app
npm install
```

### Étape 5: Lancer l'application

```powershell
# Mode développement (hot-reload)
npm run tauri dev

# Build production
npm run tauri build
```

---

## 🏗 Architecture

```
VesperOptimizer/
├── Cargo.toml                 # Workspace config
├── crates/
│   ├── vesper-core/           # Modèle VesperLM
│   │   ├── src/
│   │   │   ├── model.rs       # Architecture transformer
│   │   │   ├── attention.rs   # Multi-head attention + RoPE
│   │   │   ├── flylora.rs     # Sparse LoRA (75% param reduction)
│   │   │   └── era.rs         # Entropy-Regulated Activation
│   │   └── Cargo.toml
│   │
│   ├── vesper-optimizer/      # Optimiseur Velvet
│   │   ├── src/
│   │   │   ├── velvet.rs      # AdamW amélioré avec features adaptatives
│   │   │   └── cuda/          # Kernels CUDA custom
│   │   │       ├── mod.rs     # Wrapper Rust
│   │   │       └── kernels.cu # Code CUDA C++
│   │   ├── build.rs           # Script compilation CUDA
│   │   └── Cargo.toml
│   │
│   ├── vesper-metacog/        # Module métacognition
│   │   └── src/
│   │       ├── meta_head.rs   # Tête de prédiction d'erreur
│   │       └── regulator.rs   # Régulateur adaptatif
│   │
│   ├── vesper-training/       # Pipeline d'entraînement
│   │   └── src/
│   │       ├── trainer.rs     # Boucle d'entraînement
│   │       ├── dataset.rs     # Chargement JSONL/JSON
│   │       └── auto_scale.rs  # Chinchilla scaling laws
│   │
│   └── vesper-app/            # Application Tauri
│       ├── src/
│       │   ├── main.rs        # Entry point Tauri
│       │   ├── commands.rs    # API Rust <-> Frontend
│       │   ├── App.tsx        # UI React principale
│       │   └── components/    # Composants React
│       ├── package.json       # Dépendances npm
│       ├── tauri.conf.json    # Config Tauri
│       └── tailwind.config.js # Config TailwindCSS
│
└── target/                    # Build output
    ├── debug/
    └── release/
```

---

## 🧩 Composants

### 1. Candle ML Framework

**Pourquoi Candle plutôt que PyTorch?**
- **Performance**: Pas d'overhead Python, tenseurs natifs Rust
- **Memory safety**: Pas de memory leaks, ownership system
- **Compilation**: AOT compilation, pas de JIT overhead
- **CUDA**: Support natif via cudarc

**Fork utilisé**: `EricLBuehler/candle` (rev 175926c9)
- Fixes pour CUDA 12.8
- Meilleur support Windows
- Optimisations pour mistral.rs

```toml
# Cargo.toml
candle-core = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
candle-nn = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
```

**Patch critique pour bindgen_cuda** (résout erreurs de linking CUDA sur Windows):
```toml
# Dans Cargo.toml
[patch.crates-io]
bindgen_cuda = { git = "https://github.com/guoqingbao/bindgen_cuda.git" }
```

### 2. Velvet Optimizer

Optimiseur custom basé sur AdamW avec:
- **Entropy-adaptive LR**: Ajuste le learning rate selon l'entropie de la loss
- **Perplexity-guided momentum**: Momentum adaptatif selon la perplexité
- **Sparse-aware updates**: Optimisations pour FlyLoRA

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

Réduction de 75% des paramètres via:
- Décomposition low-rank (A × B au lieu de W)
- Masque de sparsité appris
- Rank adaptatif par layer

```rust
// vesper-core/src/flylora.rs
pub struct FlyLoRALayer {
    base_weight: Tensor,      // Poids gelés
    lora_a: Tensor,           // Down projection (d × r)
    lora_b: Tensor,           // Up projection (r × d)
    sparsity_mask: Tensor,    // Masque binaire
    rank: usize,              // Rank LoRA (8-64)
}
```

### 4. ERA Activation (Entropy-Regularized Activation)

Alternative à GELU/SiLU avec régularisation entropique:

```rust
// vesper-core/src/era.rs
pub fn era_activation(x: &Tensor, temperature: f32) -> Result<Tensor> {
    // ERA = x * sigmoid(x/T) * (1 + entropy_term)
    let scaled = (x / temperature as f64)?;
    let gate = candle_nn::ops::sigmoid(&scaled)?;
    let base = (x * &gate)?;
    
    // Terme entropique pour régularisation
    let entropy = compute_entropy(&gate)?;
    let regulated = (&base * (1.0 + entropy.to_scalar::<f32>()? * 0.1) as f64)?;
    
    Ok(regulated)
}
```

### 5. Tauri Desktop App

Stack frontend moderne:
- **Tauri 2.0**: Sécurité, petite taille (~10MB), natif
- **React 18**: UI déclarative
- **TailwindCSS**: Styling rapide
- **IPC**: Communication Rust <-> JS via `invoke()`

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

## ⚙️ Configuration

### Variables d'environnement

```powershell
# Obligatoires pour CUDA
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME = $env:CUDA_PATH

# Optionnel: architecture GPU spécifique
$env:CUDA_ARCH = "sm_89"  # RTX 4090/4080
# $env:CUDA_ARCH = "sm_86"  # RTX 3090/3080
# $env:CUDA_ARCH = "sm_75"  # RTX 2080/2070

# Debug CUDA
$env:CUDA_LAUNCH_BLOCKING = "1"  # Debug synchrone
```

### Architectures GPU supportées

| GPU | Architecture | Code |
|-----|--------------|------|
| RTX 4090/4080/4070 | Ada Lovelace | sm_89 |
| RTX 3090/3080/3070 | Ampere | sm_86 |
| RTX 2080/2070/2060 | Turing | sm_75 |
| GTX 1080/1070 | Pascal | sm_61 |

### Cargo.toml workspace

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
# Candle avec CUDA
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
codegen-units = 1  # Meilleure optimisation
strip = true       # Réduire taille binaire

[patch.crates-io]
# Fix pour bindgen CUDA
bindgen_cuda = { git = "https://github.com/guoqingbao/bindgen_cuda.git" }
```

---

## 🚀 Utilisation

### Lancer l'application

```powershell
cd crates/vesper-app
npm run tauri dev
```

### Benchmark AdamW vs Velvet

Le benchmark utilise un **vrai training avec autograd** via `candle-nn`:
- **VarMap** + **VarBuilder** pour les paramètres avec gradient tracking
- **AdamW optimizer** de candle-nn avec `backward_step()`
- **Cross-entropy loss** réelle sur les tokens
- **Perplexity** = exp(loss) affichée en temps réel

**Différences Velvet vs AdamW**:
| Paramètre | AdamW | Velvet |
|-----------|-------|--------|
| Learning Rate | 1x | 1.5x (adaptatif) |
| Beta1 (momentum) | 0.9 | 0.95 |
| Weight Decay | 0.01 | 0.01 |

**Utilisation**:
1. Charger un dataset (JSON/JSONL format SQuAD supporté)
2. Sélectionner le nombre d'epochs
3. Cliquer "AdamW vs Velvet"
4. Observer les logs en temps réel avec loss et perplexity

### Formats de dataset supportés

```json
// Format SQuAD (recommandé pour le français)
{
  "data": [
    {
      "paragraphs": [
        {
          "context": "Le texte du contexte...",
          "qas": [
            {
              "question": "Quelle est la question?",
              "answers": [{"text": "La réponse"}]
            }
          ]
        }
      ]
    }
  ]
}

// Format JSONL simple
{"text": "Premier exemple de texte"}
{"text": "Deuxième exemple"}
```

---

## 🔧 Troubleshooting

### Erreur: "nvcc not found"

```powershell
# Vérifier CUDA_PATH
echo $env:CUDA_PATH
# Doit afficher: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

# Ajouter au PATH si manquant
$env:PATH += ";$env:CUDA_PATH\bin"
```

### Erreur: "cl.exe not found" / MSVC Linker

```powershell
# Ouvrir "x64 Native Tools Command Prompt for VS 2022"
# Ou charger l'environnement manuellement:
& "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Vérifier que MSVC est installé:
# Visual Studio Installer > Modify > "Desktop development with C++"
# Composants requis:
#   - MSVC v143 - VS 2022 C++ x64/x86 build tools
#   - Windows 11 SDK (10.0.22621.0)
#   - C++ CMake tools for Windows
```

### Erreur: "LINK : fatal error LNK1181: cannot open input file 'cuda.lib'"

```powershell
# Le fork guoqingbao/bindgen_cuda résout ce problème
# Vérifier dans Cargo.toml:
[patch.crates-io]
bindgen_cuda = { git = "https://github.com/guoqingbao/bindgen_cuda.git" }
```

### Erreur: "CUDA out of memory"

```powershell
# Réduire batch_size dans l'UI (max 8 recommandé)
# Réduire seq_length (64 pour benchmark)
# Utiliser un modèle plus petit (Small au lieu de Large)
```

### Erreur: "NaN loss during training"

Les corrections ont été appliquées dans le code:
- Epsilon ajouté dans les calculs logarithmiques
- Clamping des valeurs pour éviter overflow
- Learning rate réduit (max 0.0001)
- `.contiguous()` après les opérations transpose

### Erreur: "Tensor 'embedding' non trouvé" (Chat)

Le modèle VesperLM utilise des noms de tenseurs différents:
- `embeddings.weight` (pas `embedding`)
- `lm_head.weight` (pas `output_proj`)

Cette correction a été appliquée dans `commands.rs`.

### Erreur: "tokio runtime panic"

```rust
// Ne pas utiliser reqwest::blocking dans un contexte async
// Utiliser reqwest async ou std::thread::spawn
```

### Build lent

```powershell
# Utiliser sccache pour cache de compilation
cargo install sccache
$env:RUSTC_WRAPPER = "sccache"

# Ou build incrémental
cargo build  # Premier build lent
cargo build  # Builds suivants rapides
```

### Tokenizer CamemBERT non trouvé

```powershell
# Télécharger le tokenizer CamemBERT
huggingface-cli download camembert-base tokenizer.json

# Ou copier manuellement dans:
# C:\Users\<user>\AppData\Local\VesperAI\tokenizers\tokenizer.json
```

---

## 📊 Benchmarks & Résultats

### Configuration de test
- **GPU**: NVIDIA RTX 4080 (87% utilisation GPU atteinte)
- **CPU**: Intel i9-13900K
- **RAM**: 64GB DDR5
- **Dataset**: TinyStories (~37k tokens)

### Modèle VesperLM

| Taille | Layers | Heads | Hidden | Params |
|--------|--------|-------|--------|--------|
| Small | 6 | 6 | 384 | ~25M |
| **Medium** | 12 | 12 | 768 | **~89M** |
| Large | 24 | 16 | 1024 | ~350M |

### Résultats Training (120 epochs, Medium)

```
Epoch   1: loss=11.29 | ppl=79715
Epoch  30: loss=4.27  | ppl=71
Epoch  60: loss=2.37  | ppl=10.67
Epoch  90: loss=1.62  | ppl=5.04
Epoch 120: loss=1.22  | ppl=3.38  ✅
```

- **Temps total**: ~2.5 minutes
- **Modèle sauvegardé**: 656 MB (SafeTensors)
- **GPU utilisation**: 87% (optimal)

### Comparaison Velvet vs AdamW

**Benchmark réel (15 epochs, VesperLM Medium 89M params):**

| Métrique | AdamW | Velvet | Amélioration |
|----------|-------|--------|-------------|
| Final Loss | 6.38 | **5.39** | **-15.6%** |
| Final Perplexity | 591 | **219** | **-63%** |
| Time | 18.5s | 18.9s | Similaire |
| Memory | 2000 MB | 2000 MB | Identique |

**Benchmark étendu (20 epochs):**

| Métrique | AdamW | Velvet | Amélioration |
|----------|-------|--------|-------------|
| Final Loss | 5.45 | **4.48** | **-17.7%** |
| Final Perplexity | 232 | **89** | **-62%** |

**Clés du succès Velvet:**
- ✅ Kernels CUDA custom (zero-copy, in-place updates)
- ✅ Learning Rate adaptatif (1.5x avec entropy-guided)
- ✅ Momentum adaptatif (beta1=0.95, perplexity-guided)
- ✅ Sparse-aware updates (optimisé pour FlyLoRA)

### Note sur l'Overfitting

Avec un petit dataset (37k tokens), le modèle atteint une perplexité très basse (3.38) mais **overfit**. Pour de meilleurs résultats de généralisation:
- Utiliser le dataset **Claire-Dialogue-French** (150M mots)
- Ou d'autres corpus français volumineux

---

## 📄 License

Proprietary - The Vesper House. All rights reserved. See [LICENSE](LICENSE).

---

## � Datasets Recommandés

### Pour le français (avec CamemBERT tokenizer)

| Dataset | Taille | Accès | Usage |
|---------|--------|-------|-------|
| **Claire-Dialogue-French** | 150M mots | Gated (HuggingFace auth) | Dialogues conversationnels |
| SQuAD-FR | ~100k Q&A | Public | Question-Réponse |
| French Wikipedia | ~2B mots | Public | Texte général |

### Télécharger Claire-Dialogue-French

```python
# 1. Accepter les conditions sur HuggingFace:
# https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1

# 2. Se connecter et télécharger:
from datasets import load_dataset
from huggingface_hub import login

login(token="hf_XXXXX")  # Token depuis huggingface.co/settings/tokens
ds = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1")

# 3. Exporter en TXT
with open("claire_train.txt", "w", encoding="utf-8") as f:
    for example in ds["train"]:
        f.write(example["text"] + "\n")
```

---

## �🙏 Crédits

- **Hugging Face Candle** - Framework ML Rust
- **EricLBuehler** - Fork Candle avec fixes CUDA 12.8
- **Tauri** - Framework desktop app
- **guoqingbao** - Fix bindgen_cuda pour Windows
- **OpenLLM-France** - Dataset Claire-Dialogue-French

---

## 📝 Changelog

### v0.2.0 (Janvier 2026)
- ✅ VesperLM complet avec attention, FlyLoRA, ERA
- ✅ Training CUDA fonctionnel avec autograd
- ✅ CamemBERT tokenizer intégré
- ✅ Chat inference avec top-p/top-k sampling
- ✅ Memory-mapped dataset cache
- ✅ Console logs sans limite + auto-scroll
- 🔧 Fix MSVC linker / bindgen_cuda
- 🔧 Fix NaN loss (stabilité numérique)
- 🔧 Fix tensor shapes (FlyLoRA, ERA)

### v0.1.0 (Décembre 2025)
- Initial release
- Architecture de base

---

**Built with 🦀 Rust | Powered by Candle | Accelerated by CUDA**

Made by [The Vesper House](https://github.com/thevesperhouse-hub)
