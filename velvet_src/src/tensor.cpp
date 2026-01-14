#include "suca/ml/tensor.h"
#include <cstring>
#include <cmath>
#include <random>
#include <sstream>
#include <iostream>

// Forward declarations pour CUDA (implémenté dans cuda_stubs.cpp)
namespace suca { namespace ml { namespace cuda {
    void* cuda_malloc(size_t size);
    void cuda_free(void* ptr);
    void cuda_memset(void* ptr, int value, size_t size);
    void cuda_memcpy_h2d(void* dst, const void* src, size_t size);
}}}

namespace suca {
namespace ml {

// ============================================================================
// Constructors & Destructor
// ============================================================================

Tensor::Tensor() 
    : shape_({}), device_(DeviceTypeEnum::CPU), dtype_(DType::FLOAT32), 
      data_(nullptr), requires_grad_(false), grad_(nullptr), ctx_(nullptr) {
}

Tensor::Tensor(const Shape& shape, DeviceTypeEnum device, DType dtype)
    : shape_(shape), device_(device), dtype_(dtype), 
      data_(nullptr), requires_grad_(false), grad_(nullptr), ctx_(nullptr) {
    allocate();
}

Tensor::~Tensor() {
    deallocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(other.shape_), device_(other.device_), dtype_(other.dtype_),
      data_(other.data_), requires_grad_(other.requires_grad_),
      grad_(std::move(other.grad_)), ctx_(std::move(other.ctx_)) {
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        
        shape_ = other.shape_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        data_ = other.data_;
        requires_grad_ = other.requires_grad_;
        grad_ = std::move(other.grad_);
        ctx_ = std::move(other.ctx_);
        
        other.data_ = nullptr;
    }
    return *this;
}

// ============================================================================
// Memory Management
// ============================================================================

void Tensor::allocate() {
    if (numel() == 0) return;
    
    size_t size = numel() * sizeof(float);
    
    if (device_ == DeviceTypeEnum::CPU) {
        data_ = new float[numel()];
        std::memset(data_, 0, size);
    } else {
        // Use cuda_utils for GPU allocation
        data_ = (float*)cuda::cuda_malloc(size);
        cuda::cuda_memset(data_, 0, size);
    }
}

void Tensor::deallocate() {
    if (data_ == nullptr) return;
    
    if (device_ == DeviceTypeEnum::CPU) {
        delete[] data_;
    } else {
        // Use cuda_utils for GPU deallocation
        cuda::cuda_free(data_);
    }
    
    data_ = nullptr;
}

void Tensor::copy_data_from(const float* src, size_t count) {
    if (device_ == DeviceTypeEnum::CPU) {
        std::memcpy(data_, src, count * sizeof(float));
    } else {
        // Use cuda_utils for GPU memory copy
        cuda::cuda_memcpy_h2d(data_, src, count * sizeof(float));
    }
}

// ============================================================================
// Factory Methods
// ============================================================================

Tensor Tensor::zeros(Shape shape, DeviceTypeEnum device) {
    return Tensor(shape, device);  // Already initialized to zero
}

Tensor Tensor::sum(int dim) const {
    // TODO: Implement proper reduction
    Tensor ones_tensor = Tensor::ones(shape_, DeviceTypeEnum::CPU);  // Already initialized to zero
    return ones_tensor;
}

Tensor Tensor::mean(int dim) const {
    // TODO: Implement proper reduction
    Tensor zeros_tensor = Tensor::zeros(shape_, DeviceTypeEnum::CPU);  // Already initialized to zero
    return zeros_tensor;
}

Tensor Tensor::ones(Shape shape, DeviceTypeEnum device) {
    Tensor t(shape, device);
    
    if (device == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < t.numel(); i++) {
            t.data_[i] = 1.0f;
        }
    } else {
        // TODO: CUDA kernel for ones
        throw std::runtime_error("CUDA ones not implemented yet");
    }
    
    return t;
}

Tensor Tensor::randn(Shape shape, DeviceTypeEnum device) {
    Tensor t(shape, device);
    
    if (device == DeviceTypeEnum::CPU) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int64_t i = 0; i < t.numel(); i++) {
            t.data_[i] = dist(gen);
        }
    } else {
        // TODO: CUDA kernel for randn
        throw std::runtime_error("CUDA randn not implemented yet");
    }
    
    return t;
}

Tensor Tensor::from_data(float* data, Shape shape, DeviceTypeEnum device) {
    Tensor t(shape, device);
    //t.copy_data_from(data, shape.numel());
    return t;  // TODO: implement copy_data_from
}

// ============================================================================
// Data Access
// ============================================================================

float Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() only works for scalar tensors");
    }
    
    if (device_ == DeviceTypeEnum::CPU) {
        return data_[0];
    } else {
#ifdef __CUDACC__
        float value;
        cudaMemcpy(&value, data_, sizeof(float), cudaMemcpyDeviceToHost);
        return value;
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
}

// ============================================================================
// Device Management
// ============================================================================

Tensor Tensor::to(DeviceTypeEnum target_device) const {
    if (target_device == device_) {
        return clone();
    }
    
    Tensor result(shape_, target_device, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU && target_device == DeviceTypeEnum::CUDA) {
#ifdef __CUDACC__
        cudaMemcpy(result.data_, data_, numel() * sizeof(float), cudaMemcpyHostToDevice);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else if (device_ == DeviceTypeEnum::CUDA && target_device == DeviceTypeEnum::CPU) {
#ifdef __CUDACC__
        cudaMemcpy(result.data_, data_, numel() * sizeof(float), cudaMemcpyDeviceToHost);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    
    return result;
}

// ============================================================================
// Basic Operations (CPU only for now)
// ============================================================================

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch in addition");
    }
    
    Tensor result(shape_, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < numel(); i++) {
            result.data_[i] = data_[i] + other.data_[i];
        }
    } else {
        // TODO: CUDA kernel
        throw std::runtime_error("CUDA operations not implemented yet");
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch in subtraction");
    }
    
    Tensor result(shape_, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < numel(); i++) {
            result.data_[i] = data_[i] - other.data_[i];
        }
    } else {
        throw std::runtime_error("CUDA operations not implemented yet");
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch in multiplication");
    }
    
    Tensor result(shape_, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < numel(); i++) {
            result.data_[i] = data_[i] * other.data_[i];
        }
    } else {
        throw std::runtime_error("CUDA operations not implemented yet");
    }
    
    return result;
}

// ============================================================================
// Matrix Operations
// ============================================================================

Tensor Tensor::matmul(const Tensor& other) const {
    // Simple 2D matrix multiplication for now
    if (shape_.ndim() != 2 || other.shape_.ndim() != 2) {
        throw std::runtime_error("matmul only supports 2D tensors for now");
    }
    
    int64_t m = shape_.size(0);
    int64_t k = shape_.size(1);
    int64_t n = other.shape_.size(1);
    
    if (k != other.shape_.size(0)) {
        throw std::runtime_error("Invalid dimensions for matmul");
    }
    
    Tensor result({m, n}, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        // Naive implementation (O(n^3))
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; p++) {
                    sum += data_[i * k + p] * other.data_[p * n + j];
                }
                result.data_[i * n + j] = sum;
            }
        }
    } else {
        // TODO: Use cuBLAS
        throw std::runtime_error("CUDA matmul not implemented yet");
    }
    
    return result;
}

// ============================================================================
// Utilities
// ============================================================================

Tensor Tensor::clone() const {
    Tensor result(shape_, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        std::memcpy(result.data_, data_, numel() * sizeof(float));
    } else {
#ifdef __CUDACC__
        cudaMemcpy(result.data_, data_, numel() * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
    }
    
    return result;
}

Tensor Tensor::detach() const {
    Tensor result = clone();
    result.requires_grad_ = false;
    return result;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << shape_.to_string() 
        << ", device=" << (device_ == DeviceTypeEnum::CPU ? "CPU" : "CUDA")
        << ", dtype=float32"
        << ", requires_grad=" << (requires_grad_ ? "True" : "False")
        << ")";
    return oss.str();
}

// ============================================================================
// Autograd (Placeholder)
// ============================================================================

Tensor Tensor::transpose() const {
    // For 2D matrices only
    if (shape_.ndim() != 2) {
        throw std::runtime_error("transpose() only supports 2D tensors");
    }
    
    int64_t m = shape_.size(0);
    int64_t n = shape_.size(1);
    
    Tensor result({n, m}, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                result.data_[j * m + i] = data_[i * n + j];
            }
        }
    }
    
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < numel(); i++) {
            result.data_[i] = std::max(0.0f, data_[i]);
        }
    }
    
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < numel(); i++) {
            result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
        }
    }
    
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(shape_, device_, dtype_);
    
    if (device_ == DeviceTypeEnum::CPU) {
        for (int64_t i = 0; i < numel(); i++) {
            result.data_[i] = std::tanh(data_[i]);
        }
    }
    
    return result;
}

void Tensor::backward() {
    // TODO: Implement autograd
    throw std::runtime_error("Autograd not implemented yet");
}

void Tensor::zero_grad() {
    if (grad_) {
        if (grad_->device_ == DeviceTypeEnum::CPU) {
            std::memset(grad_->data_, 0, grad_->numel() * sizeof(float));
        } else {
#ifdef __CUDACC__
            cudaMemset(grad_->data_, 0, grad_->numel() * sizeof(float));
#endif
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.to_string();
    return os;
}

} // namespace ml
} // namespace suca
