//! Velvet Optimizer - High-performance CUDA optimizer
//! 
//! Rust wrapper around custom CUDA kernels for Adam-style optimization
//! with adaptive features (entropy, perplexity, sparse-aware)

pub mod velvet;

pub use velvet::{VelvetOptimizer, VelvetConfig};

#[cfg(feature = "cuda")]
mod cuda;


/// Velvet optimizer version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
