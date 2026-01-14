# 🚀 Setup Guide - VesperAI Rust Edition

## ✅ Prerequisites Verified

- [x] **Rust 1.75+**: `rustup --version`
- [x] **CUDA 12.8**: `nvcc --version`
- [x] **Visual Studio 2019+**: Required for CUDA compilation on Windows

---

## 📦 Installation Steps

### 1. Clone & Navigate
```bash
cd F:/VelvetOptimizer
```

### 2. Build Rust Workspace (CPU only first)
```bash
# Check compilation without CUDA
cargo check --workspace --no-default-features

# Full check with CUDA (will take longer)
cargo check --workspace --all-features
```

### 3. Install Frontend Dependencies (Tauri)
```bash
cd crates/vesper-app
npm install
```

### 4. Test Examples
```bash
# Basic training example (CPU)
cargo run --example train_basic --no-default-features

# With CUDA
cargo run --example train_basic --all-features
```

### 5. Run Tauri App
```bash
cd crates/vesper-app
npm run tauri dev
```

---

## 🐛 Common Issues

### **Issue 1: CUDA Compilation Fails**
```
error: linking with `link.exe` failed
```

**Fix**:
- Ensure Visual Studio Build Tools are installed
- Set environment variable: `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- Restart terminal

### **Issue 2: Candle Not Found**
```
error: could not find `candle_core` in dependencies
```

**Fix**:
```bash
cargo clean
cargo update
cargo build
```

### **Issue 3: npm Install Fails**
```
npm ERR! code ERESOLVE
```

**Fix**:
```bash
cd crates/vesper-app
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

### **Issue 4: Tauri Build Fails**
```
error: failed to bundle project
```

**Fix**:
- Ensure Node.js 18+ is installed
- Check `tauri.conf.json` syntax
- Try: `npm run build` first, then `npm run tauri dev`

---

## 🔧 Development Workflow

### **Quick Iteration (CPU only)**
```bash
# Faster compilation without CUDA
cargo check --workspace --no-default-features
cargo run --example train_basic --no-default-features
```

### **Full Build (GPU)**
```bash
# With CUDA optimizations
cargo build --release --all-features

# Run examples with GPU
cargo run --release --example train_basic --all-features
```

### **Tauri Development**
```bash
cd crates/vesper-app

# Frontend only (hot reload)
npm run dev

# Full Tauri app (Rust + Frontend)
npm run tauri dev
```

---

## 📊 Performance Benchmarks

After successful build, run:

```bash
cargo bench
```

Expected results (RTX 4080):
- Forward pass: ~1.2ms
- Backward pass: ~2.4ms
- Optimizer step: ~0.5ms

---

## 🎯 Next Steps

1. ✅ Verify CUDA compilation
2. ✅ Test basic examples
3. ⬜ Train small model (10M params)
4. ⬜ Integrate mistral.rs for inference
5. ⬜ Add multi-GPU support

---

## 📞 Support

If you encounter issues:
1. Check `cargo --version` (should be 1.75+)
2. Check `nvcc --version` (should be CUDA 12.x)
3. Check `node --version` (should be 18+)
4. Review error logs in `target/` directory

---

**Status**: Setup in progress ⏳
