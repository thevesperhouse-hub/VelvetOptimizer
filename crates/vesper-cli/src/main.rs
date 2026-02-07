//! VesperAI CLI - Train and benchmark language models
//!
//! Usage:
//!   vesper train --dataset data.txt --tokenizer gpt2 --model-size medium --epochs 10
//!   vesper train --streaming --dataset data.txt --tokenizer gpt2 --model-size large
//!   vesper cache build --dataset data.txt --tokenizer gpt2
//!   vesper benchmark --dataset data.txt --tokenizer gpt2 --epochs 5
//!   vesper generate --model checkpoint.safetensors --tokenizer gpt2 --prompt "Hello"

mod tokenizer;
mod train;
mod benchmark;
mod generate;
mod cache;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "vesper",
    about = "VesperAI CLI - Train and benchmark language models",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a VesperLM model
    Train {
        /// Path to dataset (text or JSONL file)
        #[arg(long)]
        dataset: PathBuf,

        /// Dataset format: text, jsonl, auto
        #[arg(long, default_value = "auto")]
        format: String,

        /// HuggingFace tokenizer name (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,

        /// Model size: tiny, small, medium, large, xlarge
        #[arg(long, default_value = "small")]
        model_size: String,

        /// Number of training epochs
        #[arg(long, default_value = "3")]
        epochs: usize,

        /// Batch size
        #[arg(long, default_value = "4")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, default_value = "3e-4")]
        lr: f64,

        /// Sequence length (tokens per sample)
        #[arg(long, default_value = "512")]
        seq_len: usize,

        /// Save checkpoint every N optimizer steps (0 = only at end)
        #[arg(long, default_value = "500")]
        save_every: usize,

        /// Output directory for checkpoints
        #[arg(long, default_value = "checkpoints")]
        output_dir: PathBuf,

        /// Max training steps (0 = run all epochs)
        #[arg(long, default_value = "0")]
        max_steps: usize,

        /// Override vocab size (0 = auto-detect from tokenizer)
        #[arg(long, default_value = "0")]
        vocab_size: usize,

        /// Path to pre-built binary cache (.bin). Bypasses dataset loading.
        #[arg(long)]
        cache: Option<PathBuf>,

        /// Enable streaming mode for large text files (reads in chunks)
        #[arg(long)]
        streaming: bool,

        /// Chunk size in MB for streaming mode
        #[arg(long, default_value = "50")]
        chunk_mb: usize,
    },

    /// Benchmark Velvet optimizer vs AdamW
    Benchmark {
        /// Path to dataset
        #[arg(long)]
        dataset: PathBuf,

        /// HuggingFace tokenizer name
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,

        /// Model size: tiny, small, medium, large, xlarge
        #[arg(long, default_value = "small")]
        model_size: String,

        /// Number of epochs for each optimizer
        #[arg(long, default_value = "5")]
        epochs: usize,

        /// Batch size
        #[arg(long, default_value = "4")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, default_value = "3e-4")]
        lr: f64,

        /// Sequence length
        #[arg(long, default_value = "256")]
        seq_len: usize,

        /// Output path for JSON benchmark report
        #[arg(long, default_value = "benchmark_report.json")]
        output: PathBuf,
    },

    /// Generate text from a trained checkpoint
    Generate {
        /// Path to model checkpoint (.safetensors)
        #[arg(long)]
        model: PathBuf,

        /// HuggingFace tokenizer name
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,

        /// Model size (must match the trained model)
        #[arg(long, default_value = "small")]
        model_size: String,

        /// Prompt text
        #[arg(long)]
        prompt: String,

        /// Max tokens to generate
        #[arg(long, default_value = "100")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0.8")]
        temperature: f32,

        /// Override vocab size (0 = auto-detect)
        #[arg(long, default_value = "0")]
        vocab_size: usize,
    },

    /// Build and inspect dataset caches
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Build a binary cache from a dataset
    Build {
        /// Path to dataset file
        #[arg(long)]
        dataset: PathBuf,

        /// HuggingFace tokenizer name
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,

        /// Sequence length
        #[arg(long, default_value = "512")]
        seq_len: usize,

        /// Output directory for cache files
        #[arg(long, default_value = ".vesper_cache")]
        output_dir: PathBuf,
    },

    /// Show info about a binary cache
    Info {
        /// Path to cache .bin file
        #[arg(long)]
        cache: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            dataset, format, tokenizer, model_size,
            epochs, batch_size, lr, seq_len,
            save_every, output_dir, max_steps, vocab_size,
            cache, streaming, chunk_mb,
        } => {
            train::run(
                dataset, format, tokenizer, model_size,
                epochs, batch_size, lr, seq_len,
                save_every, output_dir, max_steps, vocab_size,
                cache, streaming, chunk_mb,
            )?;
        }

        Commands::Benchmark {
            dataset, tokenizer, model_size,
            epochs, batch_size, lr, seq_len, output,
        } => {
            benchmark::run(
                dataset, tokenizer, model_size,
                epochs, batch_size, lr, seq_len, output,
            )?;
        }

        Commands::Generate {
            model, tokenizer, model_size,
            prompt, max_tokens, temperature, vocab_size,
        } => {
            generate::run(
                model, None, tokenizer, model_size,
                prompt, max_tokens, temperature, vocab_size,
            )?;
        }

        Commands::Cache { action } => {
            match action {
                CacheAction::Build { dataset, tokenizer, seq_len, output_dir } => {
                    cache::build(dataset, tokenizer, seq_len, output_dir)?;
                }
                CacheAction::Info { cache: cache_path } => {
                    cache::info(cache_path)?;
                }
            }
        }
    }

    Ok(())
}
