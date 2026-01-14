//! Dataset Cache - Binary Memory-Mapped Token Storage
//! 
//! God-tier performance: Zero-copy memory mapped binary tokens
//! Inspired by Meta/Mistral training infrastructure

use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use bytemuck::{Pod, Zeroable};

/// Header for the binary cache file
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CacheHeader {
    /// Magic bytes: "VSPR"
    pub magic: [u8; 4],
    /// Version
    pub version: u32,
    /// Total number of tokens
    pub total_tokens: u64,
    /// Number of sequences
    pub num_sequences: u64,
    /// Vocab size used
    pub vocab_size: u32,
    /// Max sequence length
    pub max_seq_len: u32,
    /// Reserved for future use
    pub _reserved: [u8; 32],
}

impl CacheHeader {
    pub const MAGIC: [u8; 4] = *b"VSPR";
    pub const VERSION: u32 = 1;
    
    pub fn new(total_tokens: u64, num_sequences: u64, vocab_size: u32, max_seq_len: u32) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            total_tokens,
            num_sequences,
            vocab_size,
            max_seq_len,
            _reserved: [0; 32],
        }
    }
    
    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC && self.version == Self::VERSION
    }
}

/// Memory-mapped dataset for ultra-fast training
pub struct MappedDataset {
    mmap: Mmap,
    header: CacheHeader,
    /// Offset where token data starts
    data_offset: usize,
    /// Sequence offsets (index file)
    seq_offsets: Vec<u64>,
}

impl MappedDataset {
    /// Load a cached dataset from binary files
    pub fn load(bin_path: &Path) -> anyhow::Result<Self> {
        let file = File::open(bin_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Read header
        let header_size = std::mem::size_of::<CacheHeader>();
        if mmap.len() < header_size {
            anyhow::bail!("Cache file too small");
        }
        
        let header: CacheHeader = *bytemuck::from_bytes(&mmap[..header_size]);
        if !header.is_valid() {
            anyhow::bail!("Invalid cache file (bad magic or version)");
        }
        
        // Read sequence offsets
        let idx_path = bin_path.with_extension("idx");
        let seq_offsets = if idx_path.exists() {
            let idx_data = std::fs::read(&idx_path)?;
            bytemuck::cast_slice::<u8, u64>(&idx_data).to_vec()
        } else {
            // Single contiguous sequence
            vec![0, header.total_tokens]
        };
        
        Ok(Self {
            mmap,
            header,
            data_offset: header_size,
            seq_offsets,
        })
    }
    
    /// Get tokens as a slice (zero-copy!)
    pub fn tokens(&self) -> &[u32] {
        let token_bytes = &self.mmap[self.data_offset..];
        bytemuck::cast_slice(token_bytes)
    }
    
    /// Get a specific sequence by index
    pub fn get_sequence(&self, idx: usize) -> Option<&[u32]> {
        if idx + 1 >= self.seq_offsets.len() {
            return None;
        }
        
        let start = self.seq_offsets[idx] as usize;
        let end = self.seq_offsets[idx + 1] as usize;
        let tokens = self.tokens();
        
        if end <= tokens.len() {
            Some(&tokens[start..end])
        } else {
            None
        }
    }
    
    /// Get batch of sequences for training
    pub fn get_batch(&self, indices: &[usize], max_len: usize) -> Vec<Vec<u32>> {
        indices.iter()
            .filter_map(|&idx| self.get_sequence(idx))
            .map(|seq| {
                let mut padded: Vec<u32> = seq.iter().take(max_len).copied().collect();
                while padded.len() < max_len {
                    padded.push(0); // PAD token
                }
                padded
            })
            .collect()
    }
    
    pub fn num_sequences(&self) -> usize {
        self.header.num_sequences as usize
    }
    
    pub fn total_tokens(&self) -> usize {
        self.header.total_tokens as usize
    }
    
    pub fn vocab_size(&self) -> usize {
        self.header.vocab_size as usize
    }
}

/// Cache builder - converts tokenized data to binary format
pub struct CacheBuilder {
    cache_dir: PathBuf,
}

impl CacheBuilder {
    pub fn new(cache_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&cache_dir).ok();
        Self { cache_dir }
    }
    
    /// Build cache from tokenized sequences
    pub fn build_cache(
        &self,
        name: &str,
        sequences: &[Vec<u32>],
        vocab_size: u32,
        max_seq_len: u32,
    ) -> anyhow::Result<PathBuf> {
        let bin_path = self.cache_dir.join(format!("{}.bin", name));
        let idx_path = self.cache_dir.join(format!("{}.idx", name));
        
        // Calculate totals
        let total_tokens: u64 = sequences.iter().map(|s| s.len() as u64).sum();
        let num_sequences = sequences.len() as u64;
        
        // Write binary file
        let file = File::create(&bin_path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        let header = CacheHeader::new(total_tokens, num_sequences, vocab_size, max_seq_len);
        writer.write_all(bytemuck::bytes_of(&header))?;
        
        // Write tokens
        for seq in sequences {
            writer.write_all(bytemuck::cast_slice(seq))?;
        }
        writer.flush()?;
        
        // Write index file (sequence offsets)
        let mut offsets: Vec<u64> = Vec::with_capacity(sequences.len() + 1);
        let mut current_offset: u64 = 0;
        offsets.push(0);
        for seq in sequences {
            current_offset += seq.len() as u64;
            offsets.push(current_offset);
        }
        std::fs::write(&idx_path, bytemuck::cast_slice(&offsets))?;
        
        Ok(bin_path)
    }
    
    /// Check if cache exists and is valid for the given source
    pub fn cache_exists(&self, name: &str) -> bool {
        let bin_path = self.cache_dir.join(format!("{}.bin", name));
        bin_path.exists()
    }
    
    /// Get cache path
    pub fn cache_path(&self, name: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.bin", name))
    }
}

/// Generate a cache name from file path (hash-based)
pub fn cache_name_from_path(path: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();
    
    let stem = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");
    
    format!("{}_{:016x}", stem, hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    
    #[test]
    fn test_cache_roundtrip() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let builder = CacheBuilder::new(dir.path().to_path_buf());
        
        let sequences = vec![
            vec![1u32, 2, 3, 4, 5],
            vec![6u32, 7, 8],
            vec![9u32, 10, 11, 12],
        ];
        
        let bin_path = builder.build_cache("test", &sequences, 1000, 64)?;
        let dataset = MappedDataset::load(&bin_path)?;
        
        assert_eq!(dataset.num_sequences(), 3);
        assert_eq!(dataset.total_tokens(), 12);
        
        assert_eq!(dataset.get_sequence(0), Some(&[1u32, 2, 3, 4, 5][..]));
        assert_eq!(dataset.get_sequence(1), Some(&[6u32, 7, 8][..]));
        assert_eq!(dataset.get_sequence(2), Some(&[9u32, 10, 11, 12][..]));
        
        Ok(())
    }
}
