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

    /// Create a DatasetLoader from an already-loaded tokenizer
    pub fn from_tokenizer(tokenizer: tokenizers::Tokenizer, max_length: usize) -> Self {
        Self {
            tokenizer,
            max_length,
        }
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

    /// Load dataset from plain text file (for pre-training on raw corpora)
    ///
    /// Tokenizes the entire file, then chunks into fixed-length sequences.
    /// Labels are shifted by 1 for causal language modeling (next-token prediction).
    pub fn load_text(&self, path: impl AsRef<Path>) -> Result<Dataset> {
        let text = std::fs::read_to_string(path.as_ref())
            .context("Failed to read text file")?;

        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize text: {}", e))?;

        let all_ids = encoding.get_ids();

        if all_ids.len() < 2 {
            anyhow::bail!("Text file too short to create any training samples");
        }

        let mut samples = Vec::new();
        // seq_len + 1 because we need one extra token for the shifted label
        let chunk_size = self.max_length + 1;

        let mut offset = 0;
        while offset + chunk_size <= all_ids.len() {
            let chunk = &all_ids[offset..offset + chunk_size];

            let input_ids: Vec<u32> = chunk[..self.max_length].to_vec();
            let attention_mask: Vec<u32> = vec![1; self.max_length];
            // Labels shifted by 1: predict next token
            let labels: Vec<i64> = chunk[1..chunk_size].iter().map(|&id| id as i64).collect();

            samples.push(Sample {
                input_ids,
                attention_mask,
                labels,
            });

            offset += self.max_length; // non-overlapping chunks
        }

        if samples.is_empty() {
            anyhow::bail!(
                "Text file too short for seq_len {}. Got {} tokens, need at least {}",
                self.max_length,
                all_ids.len(),
                chunk_size
            );
        }

        println!("  Loaded {} sequences ({} tokens) from text file",
            samples.len(), all_ids.len());

        Ok(Dataset::new(samples))
    }

    /// Load dataset from SQuAD JSON format
    ///
    /// Extracts context paragraphs and QA pairs from SQuAD structure:
    /// { "data": [{ "paragraphs": [{ "context": "...", "qas": [{ "question": "...", "answers": [{"text": "..."}] }] }] }] }
    pub fn load_squad(&self, path: impl AsRef<Path>) -> Result<Dataset> {
        let file = std::fs::File::open(path.as_ref())
            .context("Failed to open SQuAD file")?;
        let reader = std::io::BufReader::new(file);
        let root: serde_json::Value = serde_json::from_reader(reader)
            .context("Failed to parse SQuAD JSON")?;

        let data = root["data"].as_array()
            .context("Missing 'data' array in SQuAD file")?;

        let mut all_text = String::new();

        for article in data {
            let paragraphs = match article["paragraphs"].as_array() {
                Some(p) => p,
                None => continue,
            };

            for para in paragraphs {
                if let Some(context) = para["context"].as_str() {
                    all_text.push_str(context);
                    all_text.push('\n');
                }

                if let Some(qas) = para["qas"].as_array() {
                    for qa in qas {
                        if let Some(question) = qa["question"].as_str() {
                            if let Some(answers) = qa["answers"].as_array() {
                                if let Some(answer) = answers.first() {
                                    if let Some(text) = answer["text"].as_str() {
                                        all_text.push_str(&format!("Q: {} A: {}\n", question, text));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Tokenize all collected text and chunk into sequences
        let encoding = self.tokenizer
            .encode(all_text, false)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize SQuAD text: {}", e))?;

        let all_ids = encoding.get_ids();
        let chunk_size = self.max_length + 1;
        let mut samples = Vec::new();
        let mut offset = 0;

        while offset + chunk_size <= all_ids.len() {
            let chunk = &all_ids[offset..offset + chunk_size];
            let input_ids: Vec<u32> = chunk[..self.max_length].to_vec();
            let attention_mask: Vec<u32> = vec![1; self.max_length];
            let labels: Vec<i64> = chunk[1..chunk_size].iter().map(|&id| id as i64).collect();

            samples.push(Sample {
                input_ids,
                attention_mask,
                labels,
            });

            offset += self.max_length;
        }

        if samples.is_empty() {
            anyhow::bail!("SQuAD file produced no training samples (too short for seq_len {})", self.max_length);
        }

        println!("  Loaded {} sequences ({} tokens) from SQuAD file",
            samples.len(), all_ids.len());

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

/// Streaming text loader for large-scale pretraining.
///
/// Reads a text file in chunks (default 50MB), tokenizes each chunk,
/// and yields mini-datasets of fixed-length sequences.
/// No global shuffle (standard for large-scale pretraining).
pub struct StreamingTextLoader {
    reader: std::io::BufReader<std::fs::File>,
    tokenizer: tokenizers::Tokenizer,
    seq_len: usize,
    chunk_bytes: usize,
    /// Leftover token IDs from previous chunk (incomplete sequence)
    leftover: Vec<u32>,
    exhausted: bool,
}

impl StreamingTextLoader {
    /// Create a new streaming loader.
    ///
    /// - `path`: path to a text file
    /// - `tokenizer`: pre-loaded tokenizer
    /// - `seq_len`: sequence length for each sample
    /// - `chunk_mb`: how many MB to read per chunk (default 50)
    pub fn new(
        path: impl AsRef<Path>,
        tokenizer: tokenizers::Tokenizer,
        seq_len: usize,
        chunk_mb: usize,
    ) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .context("Failed to open text file for streaming")?;
        let reader = std::io::BufReader::with_capacity(1024 * 1024, file);

        Ok(Self {
            reader,
            tokenizer,
            seq_len,
            chunk_bytes: chunk_mb * 1024 * 1024,
            leftover: Vec::new(),
            exhausted: false,
        })
    }

    /// Read next chunk and return a Dataset, or None if file exhausted.
    pub fn next_chunk(&mut self) -> Result<Option<Dataset>> {
        if self.exhausted {
            return Ok(None);
        }

        // Read chunk_bytes worth of text
        let mut buf = vec![0u8; self.chunk_bytes];
        let mut total_read = 0;

        loop {
            let n = std::io::Read::read(&mut self.reader, &mut buf[total_read..])?;
            if n == 0 {
                self.exhausted = true;
                break;
            }
            total_read += n;
            if total_read >= self.chunk_bytes {
                break;
            }
        }

        if total_read == 0 && self.leftover.is_empty() {
            return Ok(None);
        }

        // Decode UTF-8 (lossy - handles mid-char boundaries)
        let text = String::from_utf8_lossy(&buf[..total_read]);

        // Tokenize this chunk
        let encoding = self.tokenizer
            .encode(text.as_ref(), false)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize chunk: {}", e))?;

        // Combine leftover + new tokens
        let mut all_ids = std::mem::take(&mut self.leftover);
        all_ids.extend_from_slice(encoding.get_ids());

        if all_ids.len() < 2 {
            return Ok(None);
        }

        // Chunk into sequences (seq_len + 1 for shifted labels)
        let chunk_size = self.seq_len + 1;
        let mut samples = Vec::new();
        let mut offset = 0;

        while offset + chunk_size <= all_ids.len() {
            let chunk = &all_ids[offset..offset + chunk_size];
            let input_ids: Vec<u32> = chunk[..self.seq_len].to_vec();
            let attention_mask: Vec<u32> = vec![1; self.seq_len];
            let labels: Vec<i64> = chunk[1..chunk_size].iter().map(|&id| id as i64).collect();

            samples.push(Sample {
                input_ids,
                attention_mask,
                labels,
            });

            offset += self.seq_len; // non-overlapping
        }

        // Save leftover tokens for next chunk
        self.leftover = all_ids[offset..].to_vec();

        if samples.is_empty() {
            // Not enough tokens for even one sequence, try next chunk
            if self.exhausted {
                return Ok(None);
            }
            return self.next_chunk();
        }

        Ok(Some(Dataset::new(samples)))
    }

    /// Reset to beginning of file
    pub fn reset(&mut self) -> Result<()> {
        use std::io::Seek;
        self.reader.seek(std::io::SeekFrom::Start(0))?;
        self.leftover.clear();
        self.exhausted = false;
        Ok(())
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
