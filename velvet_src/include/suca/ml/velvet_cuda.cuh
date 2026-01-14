#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cstdint>

namespace suca {
namespace ml {
namespace cuda {

// CUDA kernel for Adam update (Velvet optimized)
__global__ void velvet_adam_kernel(
    float* params,
    float* m,
    float* v,
    const float* grads,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_correction1,
    float bias_correction2,
    int64_t n
);

// CUDA kernel for sparse Adam update (FlyLoRA optimized)
__global__ void velvet_sparse_adam_kernel(
    float* params,
    float* m,
    float* v,
    const float* grads,
    const uint8_t* mask,  // 1 if active, 0 if zero
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_correction1,
    float bias_correction2,
    int64_t n
);

// Host function to launch Adam update
void velvet_adam_update_cuda(
    float* d_params,
    float* d_m,
    float* d_v,
    const float* d_grads,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_correction1,
    float bias_correction2,
    int64_t n
);

// Host function for sparse update
void velvet_sparse_adam_update_cuda(
    float* d_params,
    float* d_m,
    float* d_v,
    const float* d_grads,
    const uint8_t* d_mask,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_correction1,
    float bias_correction2,
    int64_t n
);

// NEW: Complete Velvet GPU avec toutes les features adaptatives
void velvet_complete_update_cuda(
    float* d_params,
    float* d_m,
    float* d_v,
    const float* d_grads,
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_correction1,
    float bias_correction2,
    int64_t n,
    bool entropy_adaptive,
    float entropy_lr_scale,
    bool perplexity_guided,
    float ppl_momentum_scale,
    bool sparse_aware
);

} // namespace cuda
} // namespace ml
} // namespace suca

#endif // __CUDACC__
