#pragma once

#include "suca/ml/tensor.h"
#include <vector>
#include <map>
#include <memory>

namespace suca {
namespace ml {

// HIGH-LEVEL API pour Velvet GPU
// Usage simple pour LLM training
class VelvetGPUOptimizer {
public:
    struct Config {
        float base_lr = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float weight_decay = 0.0f;
        
        // Adaptive features
        bool entropy_adaptive = false;
        bool perplexity_guided = false;
        bool sparse_aware = false;
        
        // Adaptive scalars (si features activées)
        float entropy_lr_scale = 1.0f;
        float ppl_momentum_scale = 1.0f;
    };
    
    VelvetGPUOptimizer(const std::vector<Tensor*>& parameters, const Config& config = Config());
    ~VelvetGPUOptimizer();
    
    // Zero gradients
    void zero_grad();
    
    // Step optimizer (met à jour les poids)
    void step();
    
    // Update adaptive features (pour entropy/perplexity)
    void set_entropy_scale(float scale) { config_.entropy_lr_scale = scale; }
    void set_perplexity_scale(float scale) { config_.ppl_momentum_scale = scale; }
    
    // Get stats
    int get_step() const { return global_step_; }
    float get_lr() const { return config_.base_lr; }
    
private:
    struct ParamState {
        Tensor* param_gpu;      // GPU param pointer
        Tensor* grad_gpu;       // GPU gradient pointer
        Tensor m_gpu;           // GPU momentum
        Tensor v_gpu;           // GPU variance
        int step;
    };
    
    std::vector<Tensor*> parameters_;
    std::map<Tensor*, ParamState> states_;
    Config config_;
    int global_step_;
};

} // namespace ml
} // namespace suca
