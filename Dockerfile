# =============================================
# VelvetOptimizer - Vast.ai Benchmark Image
# =============================================
# Multi-stage build: compile Rust + CUDA, then slim runtime
#
# Build:
#   docker build -t vesper .
#
# Run on Vast.ai (LLaMA tokenizer - requires HF token):
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v /data:/workspace vesper benchmark \
#     --dataset /workspace/data.txt --tokenizer meta-llama/Llama-2-7b-hf \
#     --model-size xlarge --epochs 3 --batch-size 32 --lr 5e-4 \
#     --output /workspace/results.json
#
# Streaming training on large corpus:
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v /data:/workspace vesper train \
#     --streaming --dataset /workspace/fineweb-edu.txt \
#     --tokenizer meta-llama/Llama-2-7b-hf --model-size large \
#     --epochs 1 --batch-size 16 --lr 5e-4 --output-dir /workspace/checkpoints

# ---- Stage 1: Builder ----
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set CUDA architecture for A100 (sm_80)
ENV CUDA_ARCH=sm_80
ENV CUDA_PATH=/usr/local/cuda

WORKDIR /build

# Copy workspace Cargo files first (for caching)
COPY Cargo.toml Cargo.lock ./
COPY crates/vesper-core/Cargo.toml crates/vesper-core/Cargo.toml
COPY crates/vesper-optimizer/Cargo.toml crates/vesper-optimizer/Cargo.toml
COPY crates/vesper-metacog/Cargo.toml crates/vesper-metacog/Cargo.toml
COPY crates/vesper-training/Cargo.toml crates/vesper-training/Cargo.toml
COPY crates/vesper-cli/Cargo.toml crates/vesper-cli/Cargo.toml
COPY crates/vesper-app/Cargo.toml crates/vesper-app/Cargo.toml

# Create dummy src files for dependency caching
RUN mkdir -p crates/vesper-core/src && echo "pub fn dummy() {}" > crates/vesper-core/src/lib.rs && \
    mkdir -p crates/vesper-optimizer/src && echo "pub fn dummy() {}" > crates/vesper-optimizer/src/lib.rs && \
    mkdir -p crates/vesper-metacog/src && echo "pub fn dummy() {}" > crates/vesper-metacog/src/lib.rs && \
    mkdir -p crates/vesper-training/src && echo "pub fn dummy() {}" > crates/vesper-training/src/lib.rs && \
    mkdir -p crates/vesper-cli/src && echo "fn main() {}" > crates/vesper-cli/src/main.rs && \
    mkdir -p crates/vesper-app/src && echo "fn main() {}" > crates/vesper-app/src/main.rs

# Pre-download and compile dependencies (cached layer)
RUN cargo build --release -p vesper-cli 2>/dev/null || true

# Copy actual source code
COPY crates/ crates/

# Touch src files to invalidate cache for our crates only
RUN find crates/ -name "*.rs" -exec touch {} +

# Build vesper-cli (release, with CUDA)
RUN cargo build --release -p vesper-cli

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies + Python for plotting
RUN apt-get update && apt-get install -y \
    libssl3 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python plotting deps
RUN pip3 install --no-cache-dir matplotlib numpy

# Copy binary from builder
COPY --from=builder /build/target/release/vesper /usr/local/bin/vesper

# Copy plot script
COPY scripts/plot_benchmark.py /usr/local/bin/plot_benchmark.py

# Default working directory
WORKDIR /workspace

# Verify installation
RUN vesper --version

ENTRYPOINT ["vesper"]
CMD ["--help"]
