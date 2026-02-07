#!/bin/bash
# ==============================================
# VelvetOptimizer - Vast.ai Instance Setup
# ==============================================
# Run this script on a fresh Vast.ai A100 instance.
#
# Usage:
#   bash vastai_setup.sh [HF_TOKEN]
#
# Prerequisites: nvidia/cuda:12.4.1-devel-ubuntu22.04 template

set -e

echo "=== VelvetOptimizer - Vast.ai Setup ==="
echo ""

# HF token (for LLaMA tokenizer)
if [ -n "$1" ]; then
    export HF_TOKEN="$1"
elif [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HF_TOKEN set. LLaMA tokenizer won't work."
    echo "Usage: bash vastai_setup.sh hf_your_token"
    echo ""
fi

# 1. Install Rust
echo "[1/5] Installing Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "  Rust already installed."
fi

# 2. Install Python deps
echo "[2/5] Installing Python dependencies..."
pip3 install -q datasets matplotlib numpy

# 3. Clone & build
echo "[3/5] Cloning and building VelvetOptimizer..."
cd /workspace
if [ ! -d "VelvetOptimizer" ]; then
    git clone https://github.com/thevesperhouse-hub/VelvetOptimizer.git
fi
cd VelvetOptimizer

echo "  Building with CUDA_ARCH=sm_80 (A100)..."
CUDA_ARCH=sm_80 cargo build --release -p vesper-cli
echo "  Build complete!"

# 4. Download dataset
echo "[4/5] Downloading FineWeb-Edu..."
# 100M tokens for small/medium, 1B for large/xlarge
python3 scripts/download_fineweb.py --tokens 100M --output /workspace/fineweb-100M.txt
python3 scripts/download_fineweb.py --tokens 1B --output /workspace/fineweb-1B.txt

# 5. Run benchmarks
echo "[5/5] Ready to benchmark!"
echo ""
echo "Commands:"
echo ""
echo "  # Small model (quick test)"
echo "  ./target/release/vesper benchmark \\"
echo "    --dataset /workspace/fineweb-100M.txt \\"
echo "    --tokenizer meta-llama/Llama-2-7b-hf \\"
echo "    --model-size small --epochs 3 --batch-size 16 --lr 5e-4 \\"
echo "    --output /workspace/benchmark_small.json"
echo ""
echo "  # Medium model"
echo "  ./target/release/vesper benchmark \\"
echo "    --dataset /workspace/fineweb-100M.txt \\"
echo "    --tokenizer meta-llama/Llama-2-7b-hf \\"
echo "    --model-size medium --epochs 3 --batch-size 8 --lr 3e-4 \\"
echo "    --output /workspace/benchmark_medium.json"
echo ""
echo "  # Large model"
echo "  ./target/release/vesper benchmark \\"
echo "    --dataset /workspace/fineweb-1B.txt \\"
echo "    --tokenizer meta-llama/Llama-2-7b-hf \\"
echo "    --model-size large --epochs 2 --batch-size 8 --lr 3e-4 \\"
echo "    --output /workspace/benchmark_large.json"
echo ""
echo "  # XLarge / 1B params"
echo "  ./target/release/vesper benchmark \\"
echo "    --dataset /workspace/fineweb-1B.txt \\"
echo "    --tokenizer meta-llama/Llama-2-7b-hf \\"
echo "    --model-size xlarge --epochs 2 --batch-size 4 --lr 5e-4 \\"
echo "    --output /workspace/benchmark_xlarge.json"
echo ""
echo "  # Plot results"
echo "  python3 scripts/plot_benchmark.py /workspace/benchmark_small.json"
echo ""
