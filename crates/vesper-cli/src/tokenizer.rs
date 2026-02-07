//! Configurable tokenizer loading from HuggingFace Hub or local files

use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

/// Load a tokenizer by HuggingFace model name or local file path.
///
/// Examples:
/// - `load_tokenizer("gpt2")` -> downloads GPT-2 tokenizer (vocab: 50257)
/// - `load_tokenizer("meta-llama/Llama-2-7b-hf")` -> downloads LLaMA tokenizer (vocab: 32000)
/// - `load_tokenizer("camembert-base")` -> downloads CamemBERT tokenizer (vocab: 32005)
/// - `load_tokenizer("./my_tokenizer.json")` -> loads from local file
pub fn load_tokenizer(name: &str) -> Result<Tokenizer> {
    // Try local file first
    let path = std::path::Path::new(name);
    if path.exists() && path.is_file() {
        println!("  Loading tokenizer from local file: {}", name);
        return Tokenizer::from_file(name)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from file: {}", e));
    }

    // Download from HuggingFace Hub
    println!("  Downloading tokenizer: {} ...", name);
    let api = Api::new().context("Failed to initialize HuggingFace Hub API")?;
    let repo = api.repo(Repo::new(name.to_string(), RepoType::Model));

    let tokenizer_path = repo.get("tokenizer.json")
        .with_context(|| format!("Failed to download tokenizer.json from {}", name))?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load downloaded tokenizer: {}", e))?;

    let vocab_size = tokenizer.get_vocab_size(true);
    println!("  Tokenizer loaded: {} (vocab size: {})", name, vocab_size);

    Ok(tokenizer)
}

/// Get the vocabulary size from a loaded tokenizer
pub fn vocab_size(tokenizer: &Tokenizer) -> usize {
    tokenizer.get_vocab_size(true)
}
