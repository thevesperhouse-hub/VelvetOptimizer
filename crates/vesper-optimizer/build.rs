use std::env;

fn main() {
    // Only build CUDA if feature is enabled
    if cfg!(feature = "cuda") {
        build_cuda();
    }
}

fn build_cuda() {
    // Recompile si le fichier CUDA ou l'env var change
    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Find CUDA installation
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            #[cfg(target_os = "windows")]
            return "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8".to_string();
            #[cfg(not(target_os = "windows"))]
            return "/usr/local/cuda".to_string();
        });

    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Detect GPU architecture
    let arch = detect_gpu_arch().unwrap_or("sm_75".to_string());
    println!("cargo:warning=Building for GPU architecture: {}", arch);

    // Compile CUDA kernels
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag(&format!("-arch={}", arch))
        .flag("-gencode")
        .flag(&format!("arch=compute_{},code=sm_{}", 
            arch.trim_start_matches("sm_"), 
            arch.trim_start_matches("sm_")))
        .file("src/cuda/kernels.cu")
        .compile("velvet_cuda");

    // Skip bindgen - use Candle's CUDA bindings instead
}

fn detect_gpu_arch() -> Option<String> {
    // Check env var override first
    if let Ok(arch) = env::var("CUDA_ARCH") {
        return Some(arch);
    }
    
    // Try to detect GPU via nvidia-smi
    // On Windows, nvidia-smi might not be in PATH, try CUDA_PATH fallback
    #[cfg(target_os = "windows")]
    let nvidia_smi = env::var("CUDA_PATH")
        .map(|p| format!("{}/bin/nvidia-smi.exe", p.replace("\\", "/")))
        .unwrap_or_else(|_| "nvidia-smi".to_string());
    
    #[cfg(not(target_os = "windows"))]
    let nvidia_smi = "nvidia-smi".to_string();
    
    let output = std::process::Command::new(&nvidia_smi)
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;

    let gpu_name = String::from_utf8(output.stdout).ok()?;

    // Map GPU names to architectures
    if gpu_name.contains("4090") || gpu_name.contains("4080") {
        Some("sm_89".to_string()) // Ada Lovelace
    } else if gpu_name.contains("3090") || gpu_name.contains("3080") {
        Some("sm_86".to_string()) // Ampere
    } else if gpu_name.contains("2080") || gpu_name.contains("2070") {
        Some("sm_75".to_string()) // Turing
    } else {
        Some("sm_75".to_string()) // Default to Turing
    }
}
