#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void add_one_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        data[idx] += 1.0f;
    }
}

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::fprintf(                                         \
                stderr,                                           \
                "CUDA error at %s:%d: %s\n",                      \
                __FILE__,                                         \
                __LINE__,                                         \
                cudaGetErrorString(err)                           \
            );                                                     \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while (0)

int main() {
    constexpr int n = 16;

    float host[n];
    for (int i = 0; i < n; ++i) {
        host[i] = static_cast<float>(i);
    }

    float* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device, host, n * sizeof(float), cudaMemcpyHostToDevice));

    add_one_kernel<<<1, 32>>>(device, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host, device, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));

    for (int i = 0; i < n; ++i) {
        float expected = static_cast<float>(i + 1);

        if (host[i] != expected) {
            std::fprintf(
                stderr,
                "mismatch at %d: got %f expected %f\n",
                i,
                host[i],
                expected
            );
            return EXIT_FAILURE;
        }
    }

    std::printf("native CUDA smoke test passed\n");
    return EXIT_SUCCESS;
}