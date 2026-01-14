#ifndef VELVET_CUDA_H
#define VELVET_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void velvet_complete_update_cuda(
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
);

#ifdef __cplusplus
}
#endif

#endif // VELVET_CUDA_H
