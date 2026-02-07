//! Generate subcommand - Text generation from a trained checkpoint

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarMap, VarBuilder};

use vesper_core::{VesperConfig, VesperLM};

use crate::tokenizer;

/// Run the generate subcommand
pub fn run(
    model_path: std::path::PathBuf,
    _config_path: Option<std::path::PathBuf>,
    tokenizer_name: String,
    model_size: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    vocab_size_override: usize,
) -> Result<()> {
    println!("\n=== VesperAI Text Generation ===\n");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    // Load tokenizer
    let tok = tokenizer::load_tokenizer(&tokenizer_name)?;
    let vocab_size = if vocab_size_override > 0 {
        vocab_size_override
    } else {
        tokenizer::vocab_size(&tok)
    };

    // Build model config
    let config = match model_size.as_str() {
        "tiny" => VesperConfig::tiny(),
        "small" => VesperConfig::small(),
        "medium" => VesperConfig::medium(),
        "large" => VesperConfig::large(),
        "xlarge" | "1b" => VesperConfig::xlarge(),
        _ => anyhow::bail!("Unknown model size: {}", model_size),
    };
    let config = config.with_vocab_size(vocab_size);

    // Load model from checkpoint
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = VesperLM::new(config, vb)?;

    // Load saved weights
    varmap.load(&model_path)
        .with_context(|| format!("Failed to load checkpoint: {}", model_path.display()))?;
    println!("  Model loaded from: {}", model_path.display());

    // Tokenize prompt
    let encoding = tok.encode(prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    print!("{}", prompt);

    // Autoregressive generation
    for _ in 0..max_tokens {
        let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, None)?;

        // Get logits for last position
        let seq_len = token_ids.len();
        let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        // Apply temperature
        let scaled = if temperature > 0.0 && temperature != 1.0 {
            (last_logits / temperature as f64)?
        } else {
            last_logits
        };

        // Softmax + sampling
        let probs = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;
        let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

        let next_token = if temperature == 0.0 {
            // Greedy
            probs_vec.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        } else {
            // Top-p / random sampling
            sample_token(&probs_vec)
        };

        token_ids.push(next_token);

        // Decode and print the new token
        if let Some(text) = tok.decode(&[next_token], false).ok() {
            print!("{}", text);
        }
    }

    println!("\n");
    Ok(())
}

fn sample_token(probs: &[f32]) -> u32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    let mut cumulative = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return idx as u32;
        }
    }

    (probs.len() - 1) as u32
}
