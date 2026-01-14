//! Dataset loading and preprocessing

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub labels: Vec<i64>,
}

pub struct Dataset {
    samples: Vec<Sample>,
}

impl Dataset {
    pub fn new(samples: Vec<Sample>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&Sample> {
        self.samples.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Sample> {
        self.samples.iter()
    }

    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.samples.shuffle(&mut rng);
    }
}

pub struct DatasetLoader {
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
}

impl DatasetLoader {
    pub fn new(tokenizer_path: impl AsRef<Path>, max_length: usize) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    /// Load dataset from JSONL file
    pub fn load_jsonl(&self, path: impl AsRef<Path>) -> Result<Dataset> {
        let file = std::fs::File::open(path.as_ref())
            .context("Failed to open dataset file")?;
        let reader = std::io::BufReader::new(file);

        let mut samples = Vec::new();

        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            let entry: serde_json::Value = serde_json::from_str(&line)?;

            // Extract user and assistant
            let user = entry["user"].as_str().context("Missing 'user' field")?;
            let assistant = entry["assistant"].as_str().context("Missing 'assistant' field")?;

            // Tokenize
            let text = format!("User: {}\nAssistant: {}", user, assistant);
            let encoding = self.tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize: {}", e))?;

            let mut input_ids = encoding.get_ids().to_vec();
            
            // Truncate or pad
            if input_ids.len() > self.max_length {
                input_ids.truncate(self.max_length);
            }

            let attention_mask: Vec<u32> = input_ids.iter().map(|_| 1).collect();
            
            // Pad if needed
            while input_ids.len() < self.max_length {
                input_ids.push(0);
            }

            let mut attention_mask = attention_mask;
            while attention_mask.len() < self.max_length {
                attention_mask.push(0);
            }

            // Labels = input_ids (causal LM)
            let labels: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();

            samples.push(Sample {
                input_ids,
                attention_mask,
                labels,
            });
        }

        Ok(Dataset::new(samples))
    }

    /// Count total tokens in dataset
    pub fn count_tokens(&self, path: impl AsRef<Path>) -> Result<usize> {
        let file = std::fs::File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);

        let mut total_tokens = 0;

        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            let entry: serde_json::Value = serde_json::from_str(&line)?;

            if let (Some(user), Some(assistant)) = (
                entry["user"].as_str(),
                entry["assistant"].as_str(),
            ) {
                let text = format!("User: {}\nAssistant: {}", user, assistant);
                let encoding = self.tokenizer.encode(text, true)
                    .map_err(|e| anyhow::anyhow!("Failed to tokenize: {}", e))?;
                total_tokens += encoding.len();
            }
        }

        Ok(total_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let samples = vec![
            Sample {
                input_ids: vec![1, 2, 3],
                attention_mask: vec![1, 1, 1],
                labels: vec![1, 2, 3],
            },
        ];

        let dataset = Dataset::new(samples);
        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }
}
