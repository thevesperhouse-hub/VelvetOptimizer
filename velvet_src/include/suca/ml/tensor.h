#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace suca {
namespace ml {

// Forward declarations
class AutogradContext;
class Tensor;

// Device type - keeping simple enum for now
// Using Device from device.h for advanced features
enum class DeviceTypeEnum {
    CPU,
    CUDA
};

// Data type
enum class DType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
};

// Shape management
class Shape {
public:
    Shape() = default;
    Shape(std::vector<int64_t> dims) : dims_(dims) {}
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}
    
    size_t ndim() const { return dims_.size(); }
    int64_t size(int dim) const { return dims_[dim]; }
    const std::vector<int64_t>& dims() const { return dims_; }
    
    // Total number of elements
    int64_t numel() const {
        int64_t total = 1;
        for (auto d : dims_) total *= d;
        return total;
    }
    
    bool operator==(const Shape& other) const {
        return dims_ == other.dims_;
    }
    
    bool operator!=(const Shape& other) const {
        return dims_ != other.dims_;
    }
    
    std::string to_string() const {
        std::string s = "[";
        for (size_t i = 0; i < dims_.size(); i++) {
            s += std::to_string(dims_[i]);
            if (i < dims_.size() - 1) s += ", ";
        }
        s += "]";
        return s;
    }
    
private:
    std::vector<int64_t> dims_;
};

// Tensor - Core class
class Tensor {
public:
    // Constructors
    Tensor();  // Default constructor for use in containers
    Tensor(const Shape& shape, DeviceTypeEnum device = DeviceTypeEnum::CPU, DType dtype = DType::FLOAT32);
    ~Tensor();
    
    // Disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Factory methods
    static Tensor zeros(Shape shape, DeviceTypeEnum device = DeviceTypeEnum::CPU);
    static Tensor ones(Shape shape, DeviceTypeEnum device = DeviceTypeEnum::CPU);
    static Tensor randn(Shape shape, DeviceTypeEnum device = DeviceTypeEnum::CPU);
    static Tensor from_data(float* data, Shape shape, DeviceTypeEnum device = DeviceTypeEnum::CPU);
    
    // Properties
    Shape shape() const { return shape_; }
    DeviceTypeEnum device() const { return device_; }
    DType dtype() const { return dtype_; }
    int64_t numel() const { return shape_.numel(); }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool val) { requires_grad_ = val; }
    
    // Data access
    float* data() { return data_; }
    const float* data() const { return data_; }
    float item() const; // For scalar tensors
    
    // Device management
    Tensor to(DeviceTypeEnum device) const;
    Tensor cpu() const { return to(DeviceTypeEnum::CPU); }
    Tensor cuda() const { return to(DeviceTypeEnum::CUDA); }
    
    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Element-wise
    Tensor operator/(const Tensor& other) const;
    
    // In-place operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    
    // Reduction operations
    Tensor sum(int dim = -1) const;  // -1 = sum all
    Tensor mean(int dim = -1) const;
    Tensor max(int dim = -1) const;
    Tensor min(int dim = -1) const;
    
    // Activation functions
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    
    // Autograd
    void backward();
    Tensor& grad() { 
        if (!grad_) {
            grad_ = std::make_shared<Tensor>(shape_, device_, dtype_);
        }
        return *grad_; 
    }
    const Tensor& grad() const { return *grad_; }
    bool has_grad() const { return grad_ != nullptr; }
    void set_grad(const Tensor& g) {
        grad_ = std::make_shared<Tensor>(g.clone());
    }
    void zero_grad();
    
    // Utilities
    Tensor clone() const;
    Tensor detach() const;  // Remove from autograd graph
    std::string to_string() const;
    
private:
    Shape shape_;
    DeviceTypeEnum device_;
    DType dtype_;
    float* data_;  // Raw data pointer
    bool requires_grad_;
    
    // Autograd
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<AutogradContext> ctx_;
    
    // Memory management helpers
    void allocate();
    void deallocate();
    void copy_data_from(const float* src, size_t count);
};

// Helper functions
std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace ml
} // namespace suca
