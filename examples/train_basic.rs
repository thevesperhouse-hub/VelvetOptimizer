//! Basic training example

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use vesper_core::{VesperConfig, VesperLM};
use vesper_optimizer::{VelvetConfig, VelvetOptimizer};

fn main() -> Result<()> {
    println!("🔥 VesperAI Basic Training Example\n");

    // Setup
    let device = Device::cuda_if_available(0)?;
    println!("✅ Using device: {:?}", device);

    // Create model
    println!("\n📦 Creating model...");
    let config = VesperConfig::tiny();
    println!("   Model: {}M parameters", config.total_params() as f64 / 1e6);
    
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = VesperLM::new(config.clone(), vb)?;
    println!("✅ Model created");

    // Create optimizer
    println!("\n⚡ Creating Velvet optimizer...");
    let optimizer_config = VelvetConfig::optimal();
    let mut optimizer = VelvetOptimizer::new(optimizer_config);
    println!("✅ Optimizer created (lr={:.1e})", optimizer.get_lr());

    // Dummy training loop
    println!("\n🏋️  Training (dummy data)...");
    for epoch in 0..3 {
        println!("\n[Epoch {}/3]", epoch + 1);
        
        for step in 0..10 {
            // Dummy forward pass
            let input_ids = Tensor::zeros((2, 32), DType::U32, &device)?;
            let _logits = model.forward(&input_ids, None)?;
            
            // In real training: compute loss, backward, optimizer.step()
            
            if step % 5 == 0 {
                println!("  Step {}/10 - loss: {:.4}", step + 1, 0.5 + (step as f32) * 0.01);
            }
        }
        
        println!("  ✓ Epoch {} complete - avg loss: {:.4}", epoch + 1, 0.55 - (epoch as f32) * 0.1);
    }

    println!("\n✅ Training complete!\n");
    Ok(())
}
