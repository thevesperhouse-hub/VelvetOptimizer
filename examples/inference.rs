//! Basic inference example

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use vesper_core::{VesperConfig, VesperLM};

fn main() -> Result<()> {
    println!("🤖 VesperAI Inference Example\n");

    // Setup
    let device = Device::cuda_if_available(0)?;
    println!("✅ Using device: {:?}", device);

    // Create model
    println!("\n📦 Loading model...");
    let config = VesperConfig::small();
    println!("   Model: {}M parameters", config.total_params() as f64 / 1e6);
    
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = VesperLM::new(config.clone(), vb)?;
    println!("✅ Model loaded");

    // Dummy inference
    println!("\n🔮 Running inference...");
    let input_ids = Tensor::from_slice(
        &[1u32, 2, 3, 4, 5],
        5,
        &device,
    )?;
    
    let logits = model.forward(&input_ids.unsqueeze(0)?, None)?;
    println!("   Input shape: {:?}", input_ids.dims());
    println!("   Output shape: {:?}", logits.dims());
    
    // Get predictions
    let predictions = logits.argmax(2)?;
    println!("   Predictions: {:?}", predictions.to_vec2::<u32>()?);

    println!("\n✅ Inference complete!\n");
    Ok(())
}
