//! VesperAI Training Pipeline
//! 
//! Auto-scaling, dataset loading, and training loops

pub mod auto_scale;
pub mod dataset;
pub mod trainer;

pub use auto_scale::{AutoScaler, ScalingResult};
pub use dataset::{Dataset, DatasetLoader};
pub use trainer::{Trainer, TrainerConfig};


pub const VERSION: &str = env!("CARGO_PKG_VERSION");
