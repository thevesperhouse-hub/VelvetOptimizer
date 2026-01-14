//! VesperLM Core - Candle-based LLM Architecture
//! 
//! Lightweight transformer with FlyLoRA and ERA activation
//! Built with Hugging Face Candle for maximum performance

pub mod config;
pub mod model;
pub mod flylora;
pub mod era;
pub mod attention;
pub mod dataset_cache;

pub use config::VesperConfig;
pub use model::VesperLM;
pub use flylora::FlyLoRALinear;
pub use era::ERAActivation;
pub use dataset_cache::{MappedDataset, CacheBuilder, cache_name_from_path};


/// VesperLM version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
