// Velvet GPU Optimizer - PyBind11 Bindings
// Expose VelvetOptimizerCPP to Python

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <map>

// Forward declare CUDA function
extern "C" void velvet_complete_update_cuda_wrapper(
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

    void zero_grad(const std::vector<torch::Tensor>& params) {
        for (const auto& p : params) {
            if (p.grad().defined()) {
                p.grad().zero_();
            }
        }
    }

    void step(const std::vector<torch::Tensor>& params) {
        TORCH_CHECK(at::cuda::is_available(), "CUDA is not available");

        global_step_++;

        for (auto& p : params) {
            if (!p.grad().defined()) continue;

            TORCH_CHECK(p.is_cuda(), "Parameter must be on CUDA device");
            TORCH_CHECK(p.grad().is_cuda(), "Gradient must be on CUDA device");

            auto state_iter = state_.find(p.data_ptr());
            if (state_iter == state_.end()) {
                state_[p.data_ptr()] = ParamState(p);
                state_iter = state_.find(p.data_ptr());
            }

            ParamState& state = state_iter->second;
            state.step++;

            float bias_correction1 = 1.0f - std::pow(beta1_, state.step);
            float bias_correction2 = 1.0f - std::pow(beta2_, state.step);

            c10::cuda::CUDAGuard device_guard(p.device());

            int64_t numel = p.numel();

            velvet_complete_update_cuda_wrapper(
                p.data_ptr<float>(),
                state.m.data_ptr<float>(),
                state.v.data_ptr<float>(),
                p.grad().data_ptr<float>(),
                static_cast<float>(lr_),
                static_cast<float>(beta1_),
                static_cast<float>(beta2_),
                static_cast<float>(eps_),
                static_cast<float>(weight_decay_),
                bias_correction1,
                bias_correction2,
                numel,
                entropy_adaptive_,
                static_cast<float>(entropy_lr_scale_),
                perplexity_guided_,
                static_cast<float>(ppl_momentum_scale_),
                sparse_aware_
            );
        }
    }

    void set_entropy_scale(double scale) { entropy_lr_scale_ = scale; }
    void set_perplexity_scale(double scale) { ppl_momentum_scale_ = scale; }
    int64_t get_step() const { return global_step_; }
    double get_lr() const { return lr_; }

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    bool entropy_adaptive_, perplexity_guided_, sparse_aware_;
    double entropy_lr_scale_, ppl_momentum_scale_;
    int64_t global_step_;
    std::map<void*, ParamState> state_;
};

} // namespace python
} // namespace velvet

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Velvet GPU Optimizer - High-performance CUDA optimizer for PyTorch";

    py::class_<velvet::python::VelvetOptimizerCPP>(m, "VelvetOptimizerCPP")
        .def(py::init<
            double,  // lr
            double,  // beta1
            double,  // beta2
            double,  // eps
            double,  // weight_decay
            bool,    // entropy_adaptive
            bool,    // perplexity_guided
            bool,    // sparse_aware
            double,  // entropy_lr_scale
            double   // ppl_momentum_scale
        >(),
        py::arg("lr") = 0.001,
        py::arg("beta1") = 0.9,
        py::arg("beta2") = 0.999,
        py::arg("eps") = 1e-8,
        py::arg("weight_decay") = 0.0,
        py::arg("entropy_adaptive") = false,
        py::arg("perplexity_guided") = false,
        py::arg("sparse_aware") = false,
        py::arg("entropy_lr_scale") = 1.0,
        py::arg("ppl_momentum_scale") = 1.0
        )
        .def("zero_grad", &velvet::python::VelvetOptimizerCPP::zero_grad,
            "Zero all gradients",
            py::arg("params"))
        .def("step", &velvet::python::VelvetOptimizerCPP::step,
            "Perform single optimization step",
            py::arg("params"))
        .def("set_entropy_scale", &velvet::python::VelvetOptimizerCPP::set_entropy_scale,
            "Set entropy learning rate scale",
            py::arg("scale"))
        .def("set_perplexity_scale", &velvet::python::VelvetOptimizerCPP::set_perplexity_scale,
            "Set perplexity momentum scale",
            py::arg("scale"))
        .def("get_step", &velvet::python::VelvetOptimizerCPP::get_step,
            "Get current global step")
        .def("get_lr", &velvet::python::VelvetOptimizerCPP::get_lr,
            "Get learning rate");
}
