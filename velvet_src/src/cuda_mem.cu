#include <cuda_runtime.h>
#include <cstddef>

// Real CUDA memory management implementations
// These are linked when building with CUDA

namespace suca {
namespace ml {
namespace cuda {

void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void cuda_memset(void* ptr, int value, size_t size) {
    cudaMemset(ptr, value, size);
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void cuda_memcpy_d2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void cuda_synchronize() {
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace ml
} // namespace suca
