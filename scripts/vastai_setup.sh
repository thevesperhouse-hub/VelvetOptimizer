#!/bin/bash
# ==============================================
# VelvetOptimizer - Vast.ai Instance Setup
# ==============================================
# Run this script on a fresh Vast.ai GPU instance.
# Auto-detects GPU architecture (A100, RTX 6000, etc.)
#
# Usage:
#   bash vastai_setup.sh [DATASET_SIZE]
#
# DATASET_SIZE: 100M (default), 1B, 10B
# Example:
#   bash vastai_setup.sh 1B

set -e

DATASET_SIZE="${1:-1B}"

echo "=== VelvetOptimizer - Vast.ai Setup ==="
echo ""

# 1. Detect GPU
echo "[1/6] Detecting GPU..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

# 2. Install Rust
echo "[2/6] Installing Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "  Rust already installed."
fi

# 3. Install Python deps
echo "[3/6] Installing Python dependencies..."
pip3 install -q datasets matplotlib numpy

# 4. Clone & build
echo "[4/6] Cloning and building VelvetOptimizer..."
cd /workspace
if [ ! -d "VelvetOptimizer" ]; then
    git clone https://github.com/thevesperhouse-hub/VelvetOptimizer.git
fi
cd VelvetOptimizer

# Candle auto-detects GPU compute capability via nvidia-smi
echo "  Building release (CUDA auto-detect)..."
cargo build --release -p vesper-cli
echo "  Build complete!"

# 5. Download dataset
echo "[5/6] Downloading FineWeb-Edu (${DATASET_SIZE} tokens)..."
python3 scripts/download_fineweb.py --tokens "$DATASET_SIZE" --output "/workspace/fineweb-${DATASET_SIZE}.txt"

# 6. Ready
echo "[6/6] Setup complete!"
echo ""
echo "=== Training Commands ==="
echo ""
echo "  # Quick test (small model, ~5 min)"
echo "  ./target/release/vesper train \\"
echo "    --dataset /workspace/fineweb-${DATASET_SIZE}.txt \\"
echo "    --model-size small --epochs 1 --batch-size 16 --seq-len 512 \\"
echo "    --optimizer velvet --lr 5e-4 \\"
echo "    --output-dir /workspace/checkpoints"
echo ""
echo "  # 1B model training (streaming)"
echo "  ./target/release/vesper train \\"
echo "    --streaming --chunk-mb 100 \\"
echo "    --dataset /workspace/fineweb-${DATASET_SIZE}.txt \\"
echo "    --model-size 1b --epochs 1 --batch-size 16 --seq-len 512 \\"
echo "    --optimizer velvet --lr 5e-4 \\"
echo "    --output-dir /workspace/checkpoints --save-every 500"
echo ""
echo "  # Resume from checkpoint"
echo "  ./target/release/vesper train \\"
echo "    --streaming --chunk-mb 100 \\"
echo "    --dataset /workspace/fineweb-${DATASET_SIZE}.txt \\"
echo "    --model-size 1b --epochs 1 --batch-size 16 --seq-len 512 \\"
echo "    --optimizer velvet --lr 5e-4 \\"
echo "    --output-dir /workspace/checkpoints --save-every 500 \\"
echo "    --resume /workspace/checkpoints/latest.safetensors"
echo ""
