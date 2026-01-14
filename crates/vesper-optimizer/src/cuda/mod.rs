//! CUDA FFI bindings for Velvet optimizer
//! 
//! Appelle les kernels CUDA custom via FFI

use candle_core::{Device, DType, Result, Tensor, Storage};
use candle_core::cuda_backend::CudaStorageSlice;

// Lien vers le kernel CUDA compilé par build.rs
extern "C" {
    fn velvet_complete_update_cuda(
        params: *mut f32,
        m: *mut f32,
        v: *mut f32,
        grads: *const f32,
        base_lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        n: i64,
        entropy_adaptive: bool,
        entropy_lr_scale: f32,
        perplexity_guided: bool,
        ppl_momentum_scale: f32,
        sparse_aware: bool,
    );
}

/// Appelle le kernel CUDA Velvet sur les tensors
/// 
/// Cette fonction encapsule l'appel FFI. Les tensors doivent être sur CUDA et F32.
pub fn velvet_update_cuda(
    params: &Tensor,
    m: &Tensor,
    v: &Tensor,
    grads: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    entropy_adaptive: bool,
    entropy_scale: f32,
    perplexity_guided: bool,
    perplexity_scale: f32,
    sparse_aware: bool,
) -> Result<()> {
    // Vérifier que tous les tensors sont sur CUDA et F32
    if params.dtype() != DType::F32 {
        candle_core::bail!("params must be F32");
    }
    
    let n = params.elem_count() as i64;
    
    // Obtenir le CudaDevice et son stream
    let cuda_dev = match params.device() {
        Device::Cuda(dev) => dev.clone(),
        _ => candle_core::bail!("Tensors must be on CUDA device"),
    };
    let stream = cuda_dev.cuda_stream();
    
    // Extraire les pointeurs device
    let params_ptr = get_device_ptr_mut(params, &stream)?;
    let m_ptr = get_device_ptr_mut(m, &stream)?;
    let v_ptr = get_device_ptr_mut(v, &stream)?;
    let grads_ptr = get_device_ptr(grads, &stream)?;
    
    // Appel FFI au kernel CUDA (encapsulé dans unsafe car c'est du FFI)
    // SAFETY: Les pointeurs sont valides car extraits de tensors Candle CUDA valides
    unsafe {
        velvet_complete_update_cuda(
            params_ptr,
            m_ptr,
            v_ptr,
            grads_ptr,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            bias_correction1,
            bias_correction2,
            n,
            entropy_adaptive,
            entropy_scale,
            perplexity_guided,
            perplexity_scale,
            sparse_aware,
        );
    }
    
    Ok(())
}

/// Extrait le pointeur device mutable d'un tensor CUDA F32
fn get_device_ptr_mut(tensor: &Tensor, stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>) -> Result<*mut f32> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    
    let (storage, layout) = tensor.storage_and_layout();
    let offset = layout.start_offset();
    
    match &*storage {
        Storage::Cuda(cuda_storage) => {
            match &cuda_storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let view = slice.slice(offset..);
                    let (ptr, _sync) = view.device_ptr(stream);
                    Ok(ptr as *mut f32)
                }
                _ => candle_core::bail!("Expected F32 tensor"),
            }
        }
        _ => candle_core::bail!("Expected CUDA tensor"),
    }
}

/// Extrait le pointeur device (lecture seule) d'un tensor CUDA F32
fn get_device_ptr(tensor: &Tensor, stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>) -> Result<*const f32> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    
    let (storage, layout) = tensor.storage_and_layout();
    let offset = layout.start_offset();
    
    match &*storage {
        Storage::Cuda(cuda_storage) => {
            match &cuda_storage.slice {
                CudaStorageSlice::F32(slice) => {
                    let view = slice.slice(offset..);
                    let (ptr, _sync) = view.device_ptr(stream);
                    Ok(ptr as *const f32)
                }
                _ => candle_core::bail!("Expected F32 tensor"),
            }
        }
        _ => candle_core::bail!("Expected CUDA tensor"),
    }
}
