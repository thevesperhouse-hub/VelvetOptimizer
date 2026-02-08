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
    // Check env var override first (CUDA_ARCH or CUDA_COMPUTE_CAP)
    if let Ok(arch) = env::var("CUDA_ARCH") {
        return Some(arch);
    }
    if let Ok(cap) = env::var("CUDA_COMPUTE_CAP") {
        return Some(format!("sm_{}", cap));
    }

    // Try to detect via nvidia-smi compute capability
    let nvidia_smi = get_nvidia_smi_path();
    if let Some(arch) = detect_via_compute_cap(&nvidia_smi) {
        return Some(arch);
    }
    detect_via_gpu_name(&nvidia_smi)
}

fn get_nvidia_smi_path() -> String {
    #[cfg(target_os = "windows")]
    {
        env::var("CUDA_PATH")
            .map(|p| format!("{}/bin/nvidia-smi.exe", p.replace("\\", "/")))
            .unwrap_or_else(|_| "nvidia-smi".to_string())
    }
    #[cfg(not(target_os = "windows"))]
    {
        "nvidia-smi".to_string()
    }
}

fn detect_via_compute_cap(nvidia_smi: &str) -> Option<String> {
    let output = std::process::Command::new(nvidia_smi)
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;
    let cap = String::from_utf8(output.stdout).ok()?;
    let cap = cap.trim();
    let parts: Vec<&str> = cap.split('.').collect();
    if parts.len() == 2 {
        Some(format!("sm_{}{}", parts[0], parts[1]))
    } else {
        None
    }
}

fn detect_via_gpu_name(nvidia_smi: &str) -> Option<String> {
    let output = std::process::Command::new(nvidia_smi)
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;
    let gpu_name = String::from_utf8(output.stdout).ok()?;

    if gpu_name.contains("B200") || gpu_name.contains("B100") || gpu_name.contains("Blackwell") {
        Some("sm_100".to_string())
    } else if gpu_name.contains("H100") || gpu_name.contains("H200") {
        Some("sm_90".to_string())
    } else if gpu_name.contains("A100") || gpu_name.contains("A800") {
        Some("sm_80".to_string())
    } else if gpu_name.contains("A6000") || gpu_name.contains("A5000") {
        Some("sm_86".to_string())
    } else if gpu_name.contains("4090") || gpu_name.contains("4080") || gpu_name.contains("L40") {
        Some("sm_89".to_string())
    } else if gpu_name.contains("3090") || gpu_name.contains("3080") || gpu_name.contains("3070") {
        Some("sm_86".to_string())
    } else if gpu_name.contains("2080") || gpu_name.contains("2070") {
        Some("sm_75".to_string())
    } else {
        Some("sm_80".to_string()) // Safe default
    }
}
