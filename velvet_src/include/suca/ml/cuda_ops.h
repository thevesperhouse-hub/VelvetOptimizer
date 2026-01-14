#pragma once

#include <cstdint>

namespace suca {
namespace ml {
namespace cuda {

// Element-wise operations
void launch_add(float* out, const float* a, const float* b, int64_t n);
void launch_sub(float* out, const float* a, const float* b, int64_t n);
void launch_mul(float* out, const float* a, const float* b, int64_t n);
void launch_div(float* out, const float* a, const float* b, int64_t n);

// Scalar operations
void launch_add_scalar(float* out, const float* a, float b, int64_t n);
void launch_mul_scalar(float* out, const float* a, float b, int64_t n);

// Activations
void launch_relu(float* out, const float* in, int64_t n);
void launch_relu_backward(float* grad_in, const float* grad_out, const float* in, int64_t n);
void launch_sigmoid(float* out, const float* in, int64_t n);
void launch_sigmoid_backward(float* grad_in, const float* grad_out, const float* out, int64_t n);
void launch_tanh(float* out, const float* in, int64_t n);
void launch_tanh_backward(float* grad_in, const float* grad_out, const float* out, int64_t n);

// Loss functions
void launch_mse_loss(float* loss, const float* pred, const float* target, int64_t n);
void launch_mse_loss_backward(float* grad, const float* pred, const float* target, int64_t n);

} // namespace cuda
} // namespace ml
} // namespace suca
