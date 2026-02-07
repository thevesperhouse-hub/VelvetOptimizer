//! Cache subcommand - Build and inspect binary dataset caches

use anyhow::{Context, Result};
use std::path::PathBuf;

use vesper_core::{CacheBuilder, MappedDataset, cache_name_from_path};
use vesper_training::DatasetLoader;

use crate::tokenizer;

/// Build a binary cache from a text/JSONL dataset
pub fn build(
    dataset_path: PathBuf,
    tokenizer_name: String,
    seq_len: usize,
    output_dir: PathBuf,
) -> Result<()> {
    println!("\n=== VesperAI Cache Builder ===\n");

    // Load tokenizer ("auto" defaults to gpt2 for cache — no model context)
    let resolved = if tokenizer_name == "auto" { "gpt2".to_string() } else { tokenizer_name };
    let tok = tokenizer::load_tokenizer(&resolved)?;
    let vocab_size = tokenizer::vocab_size(&tok);

    // Load and tokenize dataset
    let loader = DatasetLoader::from_tokenizer(tok, seq_len);

    let format = match dataset_path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => "jsonl",
        Some("json") => "squad",
        _ => "text",
    };

    println!("  Loading dataset: {} (format: {})", dataset_path.display(), format);
    let dataset = match format {
        "jsonl" => loader.load_jsonl(&dataset_path)?,
        "squad" | "json" => loader.load_squad(&dataset_path)?,
        _ => loader.load_text(&dataset_path)?,
    };

    println!("  Tokenized: {} sequences, seq_len={}", dataset.len(), seq_len);

    // Convert samples to Vec<Vec<u32>> for CacheBuilder
    let sequences: Vec<Vec<u32>> = dataset.iter()
        .map(|s| s.input_ids.clone())
        .collect();

    // Build cache
    let cache_name = cache_name_from_path(&dataset_path);
    let builder = CacheBuilder::new(output_dir.clone());

    println!("  Building cache: {}", cache_name);
    let cache_path = builder.build_cache(
        &cache_name,
        &sequences,
        vocab_size as u32,
        seq_len as u32,
    )?;

    println!("\n  Cache built successfully!");
    println!("  Path: {}", cache_path.display());
    println!("  Sequences: {}", sequences.len());
    println!("  Total tokens: {}", sequences.iter().map(|s| s.len()).sum::<usize>());

    Ok(())
}

/// Display info about an existing binary cache
pub fn info(cache_path: PathBuf) -> Result<()> {
    println!("\n=== Cache Info ===\n");

    let dataset = MappedDataset::load(&cache_path)
        .context("Failed to load cache file")?;

    println!("  Path: {}", cache_path.display());
    println!("  Sequences: {}", dataset.num_sequences());
    println!("  Total tokens: {}", dataset.total_tokens());
    println!("  Vocab size: {}", dataset.vocab_size());

    // File size
    let meta = std::fs::metadata(&cache_path)?;
    let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
    println!("  File size: {:.1} MB", size_mb);

    Ok(())
}
