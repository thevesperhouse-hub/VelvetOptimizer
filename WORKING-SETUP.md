# ✅ Configuration Qui Marche (AttentionR&D Style)

## 🎯 Setup Simplifié

### 1. Configuration MSVC (Une fois par session)

Dans ton terminal (PowerShell, CMD, ou VS Code terminal):

```cmd
setup-env.cmd
```

Ou manuellement:
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

**C'est tout.** Pas besoin de configurer PATH/LIB manuellement.

---

### 2. Compilation

Après avoir lancé `setup-env.cmd`:

```bash
# Clean build
cargo clean

# Check compilation
cargo check --workspace --no-default-features

# Build release
cargo build --release --all-features
```

---

## 📋 Cargo.toml Configuration

```toml
# Exactement comme AttentionR&D qui fonctionne
candle-core = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9", features = ["cuda"] }
candle-nn = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9" }
candle-transformers = { git = "https://github.com/EricLBuehler/candle.git", rev = "175926c9" }
```

**Pourquoi `rev = "175926c9"`?**
- Fix pour CUDA 12.8/12.9 bug `sm_32_intrinsics.hpp`
- Version testée et stable
- Pas de downgrade, c'est la version qui marche avec ton setup

---

## 🚀 Workflow Complet

```cmd
# 1. Setup environment (une fois par session)
setup-env.cmd

# 2. Clean + compile
cd F:\VelvetOptimizer
cargo clean
cargo check --workspace --no-default-features

# 3. Si ça compile, test avec CUDA
cargo check --workspace --all-features

# 4. Install npm dependencies pour Tauri
cd crates\vesper-app
npm install

# 5. Launch Tauri app
npm run tauri dev
```

---

## ⚠️ Troubleshooting

### Erreur: `kernel32.lib` introuvable
→ Tu n'as pas lancé `setup-env.cmd` ou `vcvars64.bat`

### Erreur: `link.exe` not found  
→ Tu n'as pas lancé `setup-env.cmd` ou `vcvars64.bat`

### Erreur: CUDA compilation fails
→ Vérifie `nvcc --version` fonctionne après `setup-env.cmd`

---

## 📌 À Retenir

**Nouveau terminal = Relancer setup-env.cmd**

C'est comme ça que ça marche dans AttentionR&D, c'est comme ça que ça va marcher ici.

Simple. Efficace. Testé.
