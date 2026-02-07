// Velvet CUDA Kernels - Ported from C++ version
// High-performance Adam-style optimizer with adaptive features

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void velvet_complete_kernel(
    float* params,
    float* m,
    float* v,
    const float* grads,
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_correction1,
    float bias_correction2,
    long long n,
    bool entropy_adaptive,
    float entropy_lr_scale,
    bool perplexity_guided,
    float ppl_momentum_scale,
    bool sparse_aware
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float p = params[idx];
        
        // Sparse optimization: skip near-zero weights
        if (sparse_aware && fabsf(p) < 1e-9f) return;
        
        float g = grads[idx];
        float m_val = m[idx];
        float v_val = v[idx];

        // Decoupled weight decay (AdamW style)
        p *= (1.0f - base_lr * wd);

        // Apply adaptive momentum (perplexity-guided)
        float effective_beta1 = beta1;
        if (perplexity_guided) {
            effective_beta1 *= ppl_momentum_scale;
            effective_beta1 = fminf(fmaxf(effective_beta1, 0.5f), 0.999f);
        }

        float one_minus_beta1 = 1.0f - effective_beta1;
        float one_minus_beta2 = 1.0f - beta2;

        // Update moments
        m_val = effective_beta1 * m_val + one_minus_beta1 * g;
        v_val = beta2 * v_val + one_minus_beta2 * g * g;

        // Bias-corrected moments
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        // Apply adaptive LR (entropy-guided)
        float effective_lr = base_lr;
        if (entropy_adaptive) {
            effective_lr *= entropy_lr_scale;
        }

        // Update parameter
        p -= effective_lr * m_hat / (sqrtf(v_hat) + eps);
        
        // Store results
        params[idx] = p;
        m[idx] = m_val;
        v[idx] = v_val;
    }
}

extern "C" void velvet_complete_update_cuda(
    float* params,
    float* m,
    float* v,
    const float* grads,
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bias_correction1,
    float bias_correction2,
    long long n,
    bool entropy_adaptive,
    float entropy_lr_scale,
    bool perplexity_guided,
    float ppl_momentum_scale,
    bool sparse_aware
) {
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    velvet_complete_kernel<<<num_blocks, threads_per_block>>>(
        params, m, v, grads,
        base_lr, beta1, beta2, eps, wd,
        bias_correction1, bias_correction2, n,
        entropy_adaptive, entropy_lr_scale,
        perplexity_guided, ppl_momentum_scale,
        sparse_aware
    );

    // No cudaDeviceSynchronize: CUDA stream ordering guarantees kernel
    // serialization. Sync only needed when CPU reads results (e.g. to_scalar).
}
