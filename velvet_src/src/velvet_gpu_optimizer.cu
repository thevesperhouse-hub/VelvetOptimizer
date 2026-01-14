#include "suca/ml/velvet_gpu_optimizer.h"
#include "suca/ml/velvet_cuda.cuh"
#include "suca/ml/cuda_utils.h"
#include <cmath>
#include <stdexcept>

namespace suca {
namespace ml {

VelvetGPUOptimizer::VelvetGPUOptimizer(const std::vector<Tensor*>& parameters, const Config& config)
    : parameters_(parameters), config_(config), global_step_(0) {
    
    // Initialize state pour chaque paramètre
    for (auto* param : parameters_) {
        if (param->device() != DeviceTypeEnum::CUDA) {
            throw std::runtime_error("VelvetGPUOptimizer: All parameters must be on CUDA device!");
        }
        
        // Use emplace pour éviter la copie
        states_.emplace(std::piecewise_construct,
                       std::forward_as_tuple(param),
                       std::forward_as_tuple());
        
        auto& state = states_.at(param);
        state.param_gpu = param;
        state.grad_gpu = nullptr;
        state.m_gpu = Tensor::zeros(param->shape(), DeviceTypeEnum::CUDA);
        state.v_gpu = Tensor::zeros(param->shape(), DeviceTypeEnum::CUDA);
        state.step = 0;
    }
}

VelvetGPUOptimizer::~VelvetGPUOptimizer() {
    // Les tensors sont gérés automatiquement par RAII
}

void VelvetGPUOptimizer::zero_grad() {
    // Zero tous les gradients
    for (auto* param : parameters_) {
        if (param->has_grad()) {
            auto& grad = param->grad();
            if (grad.device() == DeviceTypeEnum::CUDA) {
                cuda::cuda_memset(grad.data(), 0, grad.numel() * sizeof(float));
            }
        }
    }
}

void VelvetGPUOptimizer::step() {
    global_step_++;
    
    for (auto* param : parameters_) {
        if (!param->has_grad()) continue;
        
        auto& state = states_[param];
        state.step++;
        
        auto& grad = param->grad();
        if (grad.device() != DeviceTypeEnum::CUDA) {
            throw std::runtime_error("VelvetGPUOptimizer: Gradient must be on CUDA device!");
        }
        
        // Bias corrections
        float bias_correction1 = 1.0f - std::pow(config_.beta1, state.step);
        float bias_correction2 = 1.0f - std::pow(config_.beta2, state.step);
        
        // Call Velvet GPU kernel
        cuda::velvet_complete_update_cuda(
            param->data(),              // params
            state.m_gpu.data(),         // m
            state.v_gpu.data(),         // v
            grad.data(),                // grads
            config_.base_lr,            // lr
            config_.beta1,              // beta1
            config_.beta2,              // beta2
            config_.epsilon,            // eps
            config_.weight_decay,       // weight_decay
            bias_correction1,           // bias_correction1
            bias_correction2,           // bias_correction2
            param->numel(),             // n
            config_.entropy_adaptive,   // entropy_adaptive
            config_.entropy_lr_scale,   // entropy_lr_scale
            config_.perplexity_guided,  // perplexity_guided
            config_.ppl_momentum_scale, // ppl_momentum_scale
            config_.sparse_aware        // sparse_aware
        );
    }
}

} // namespace ml
} // namespace suca
