//! Metacognition Module for VesperLM
//! 
//! Lightweight metacognitive layer for error detection and confidence estimation
//! Inspired by Meta-R1 and META3

pub mod meta_head;
pub mod regulator;

pub use meta_head::{MetaHead, MetaHeadConfig};
pub use regulator::{MetacognitiveRegulator, RegulatorConfig};


pub const VERSION: &str = env!("CARGO_PKG_VERSION");
