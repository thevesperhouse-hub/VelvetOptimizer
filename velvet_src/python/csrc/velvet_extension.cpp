// Velvet GPU Optimizer - PyTorch C++ Extension
// Wrapper autour de VelvetGPUOptimizer pour interface PyTorch

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "suca/ml/velvet_cuda.cuh"
#include <vector>
#include <map>

namespace velvet {
namespace python {

// Structure pour stocker l'état d'un paramètre
struct ParamState {
    torch::Tensor m;  // Momentum
    torch::Tensor v;  // Variance
    int64_t step;

    ParamState() : step(0) {}

    ParamState(const torch::Tensor& param_shape) : step(0) {
        m = torch::zeros_like(param_shape);
        v = torch::zeros_like(param_shape);
    }
};

// Classe principale VelvetOptimizerCPP
// Interface C++ pour être appelée depuis Python
class VelvetOptimizerCPP {
public:
    VelvetOptimizerCPP(
        double lr,
        double beta1,
        double beta2,
        double eps,
        double weight_decay,
        bool entropy_adaptive,
        bool perplexity_guided,
        bool sparse_aware,
        double entropy_lr_scale,
        double ppl_momentum_scale
    )
        : lr_(lr)
        , beta1_(beta1)
        , beta2_(beta2)
        , eps_(eps)
        , weight_decay_(weight_decay)
        , entropy_adaptive_(entropy_adaptive)
        , perplexity_guided_(perplexity_guided)
        , sparse_aware_(sparse_aware)
        , entropy_lr_scale_(entropy_lr_scale)
        , ppl_momentum_scale_(ppl_momentum_scale)
        , global_step_(0)
    {}

    // Zero gradients
    void zero_grad(const std::vector<torch::Tensor>& params) {
        for (const auto& p : params) {
            if (p.grad().defined()) {
                p.grad().zero_();
            }
        }
    }

    // Step optimizer
    void step(const std::vector<torch::Tensor>& params) {
        TORCH_CHECK(at::cuda::is_available(), "CUDA is not available");

        global_step_++;

        for (auto& p : params) {
            // Skip si pas de gradient
            if (!p.grad().defined()) {
                continue;
            }

            TORCH_CHECK(p.is_cuda(), "Parameter must be on CUDA device");
            TORCH_CHECK(p.grad().is_cuda(), "Gradient must be on CUDA device");

            // Récupère ou initialise l'état pour ce paramètre
            auto state_iter = state_.find(p.data_ptr());
            if (state_iter == state_.end()) {
                // Premier step pour ce paramètre - initialise l'état
                state_[p.data_ptr()] = ParamState(p);
                state_iter = state_.find(p.data_ptr());
            }

            ParamState& state = state_iter->second;
            state.step++;

            // Bias corrections
            float bias_correction1 = 1.0f - std::pow(beta1_, state.step);
            float bias_correction2 = 1.0f - std::pow(beta2_, state.step);

            // Assure que tout est sur le même device
            c10::cuda::CUDAGuard device_guard(p.device());

            // Appelle le kernel Velvet CUDA
            int64_t numel = p.numel();

            suca::ml::cuda::velvet_complete_update_cuda(
                p.data_ptr<float>(),                    // params
                state.m.data_ptr<float>(),              // m
                state.v.data_ptr<float>(),              // v
                p.grad().data_ptr<float>(),             // grads
                static_cast<float>(lr_),                // base_lr
                static_cast<float>(beta1_),             // beta1
                static_cast<float>(beta2_),             // beta2
                static_cast<float>(eps_),               // eps
                static_cast<float>(weight_decay_),      // weight_decay
                bias_correction1,                       // bias_correction1
                bias_correction2,                       // bias_correction2
                numel,                                  // n
                entropy_adaptive_,                      // entropy_adaptive
                static_cast<float>(entropy_lr_scale_),  // entropy_lr_scale
                perplexity_guided_,                     // perplexity_guided
                static_cast<float>(ppl_momentum_scale_),// ppl_momentum_scale
                sparse_aware_                           // sparse_aware
            );
        }

        // Sync CUDA (optionnel, mais aide pour le debugging)
        // cudaDeviceSynchronize();
    }

    // Setters pour features adaptatives
    void set_entropy_scale(double scale) { entropy_lr_scale_ = scale; }
    void set_perplexity_scale(double scale) { ppl_momentum_scale_ = scale; }

    // Getters
    int64_t get_step() const { return global_step_; }
    double get_lr() const { return lr_; }

private:
    // Config
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    bool entropy_adaptive_;
    bool perplexity_guided_;
    bool sparse_aware_;
    double entropy_lr_scale_;
    double ppl_momentum_scale_;

    // State
    int64_t global_step_;
    std::map<void*, ParamState> state_;  // Map parameter data_ptr -> state
};

} // namespace python
} // namespace velvet
