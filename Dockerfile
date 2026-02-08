# =============================================
# VelvetOptimizer - GPU Training Image
# =============================================
# Multi-stage build: compile Rust + CUDA, then slim runtime
#
# Build:
#   docker build -t vesper .
#
# Run training (streaming, 1B model):
#   docker run --gpus all -v /data:/workspace vesper train \
#     --streaming --chunk-mb 100 --dataset /workspace/fineweb-1B.txt \
#     --model-size 1b --epochs 1 --batch-size 16 --seq-len 512 \
#     --optimizer velvet --lr 5e-4 --output-dir /workspace/checkpoints \
#     --save-every 500
#
# Benchmark:
#   docker run --gpus all -v /data:/workspace vesper benchmark \
#     --dataset /workspace/data.txt --model-size xlarge \
#     --epochs 3 --batch-size 16 --lr 5e-4 \
#     --output /workspace/results.json

# ---- Stage 1: Builder ----
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

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

# Build vesper-cli (release, with CUDA — auto-detects compute capability)
RUN cargo build --release -p vesper-cli

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies + Python for dataset download & plotting
RUN apt-get update && apt-get install -y \
    libssl3 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip3 install --no-cache-dir matplotlib numpy datasets

# Copy binary from builder
COPY --from=builder /build/target/release/vesper /usr/local/bin/vesper

# Copy scripts
COPY scripts/ /usr/local/share/vesper/scripts/

# Default working directory
WORKDIR /workspace

# Verify installation
RUN vesper --version

ENTRYPOINT ["vesper"]
CMD ["--help"]
