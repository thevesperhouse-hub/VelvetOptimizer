#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <stdexcept>
#include <string>
#include <iostream>

namespace suca {
namespace ml {
namespace cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + \
                cudaGetErrorString(error) \
            ); \
        } \
    } while(0)

// cuBLAS error checking
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error( \
                std::string("cuBLAS error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) \
            ); \
        } \
    } while(0)

// Memory management - declarations only (implemented in cuda_mem.cu or stubs)
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
void cuda_memcpy_h2d(void* dst, const void* src, size_t size);
void cuda_memcpy_d2h(void* dst, const void* src, size_t size);
void cuda_memcpy_d2d(void* dst, const void* src, size_t size);
void cuda_memset(void* ptr, int value, size_t size);
void cuda_synchronize();

// cuBLAS handle management
class CublasHandle {
public:
    static cublasHandle_t& get() {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            CUBLAS_CHECK(cublasCreate(&handle));
        }
        return handle;
    }
    
    static void destroy() {
        auto& handle = get();
        if (handle != nullptr) {
            cublasDestroy(handle);
            handle = nullptr;
        }
    }
};

// Device info
inline int get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

inline void set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

inline int get_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

inline void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
}

} // namespace cuda
} // namespace ml
} // namespace suca
