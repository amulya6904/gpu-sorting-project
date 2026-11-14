#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {
    // Each thread gets a unique array index.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    // This size leaves unused threads in the last block.
    int n = 1000;
    if (argc == 2 && std::string(argv[1]) == "257") n = 257;
    else if (argc > 2 || (argc == 2 && std::string(argv[1]) != "1000")) {
        std::cerr << "FAIL: Usage: vector_add [1000|257]\n";
        return EXIT_FAILURE;
    }
    constexpr int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    const size_t bytes = n * sizeof(int);

    std::vector<int> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    int *dA = nullptr, *dB = nullptr, *dC = nullptr;
    // Attempt all frees even if an earlier CUDA operation failed.
    auto cleanup = [&]() {
        bool ok = true;
        for (int** pointer : {&dA, &dB, &dC}) {
            if (*pointer) {
                cudaError_t status = cudaFree(*pointer);
                *pointer = nullptr;
                if (status != cudaSuccess) {
                    std::cerr << "FAIL: Free device memory: "
                              << cudaGetErrorString(status) << '\n';
                    ok = false;
                }
            }
        }
        return ok;
    };
    try {
        check(cudaMalloc(&dA, bytes), "Allocate GPU input A");
        check(cudaMalloc(&dB, bytes), "Allocate GPU input B");
        check(cudaMalloc(&dC, bytes), "Allocate GPU output C");
        check(cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice), "Copy A to GPU");
        check(cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice), "Copy B to GPU");

        vectorAdd<<<blocks, threadsPerBlock>>>(dA, dB, dC, n);
        check(cudaGetLastError(), "Launch vectorAdd");
        check(cudaDeviceSynchronize(), "Finish vectorAdd");

        check(cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost), "Copy C to CPU");
        if (!cleanup()) return EXIT_FAILURE;

        for (int i = 0; i < n; ++i) {
            if (c[i] != a[i] + b[i]) {
                std::cerr << "FAIL: Mismatch at index " << i << '\n';
                return EXIT_FAILURE;
            }
        }

        std::cout << "PASS: " << n << " sums verified using " << blocks
                  << " blocks of " << threadsPerBlock << " threads.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        cleanup();
        return EXIT_FAILURE;
    }
}
