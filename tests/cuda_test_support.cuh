#pragma once
#include "bitonic_sort.cuh"
#include "sorting_support.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace cuda_test {
inline void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct Resources {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    std::uint32_t* keys = nullptr;
    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    void initialize(std::size_t capacity) {
        int count = 0;
        check(cudaGetDeviceCount(&count), "Query CUDA devices");
        if (count == 0) throw std::runtime_error("No CUDA GPU; verification pending");
        check(cudaSetDevice(0), "Select GPU");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "Read GPU properties");
        std::cout << "GPU: " << prop.name << '\n';
        if (prop.maxThreadsPerBlock < 256 || prop.sharedMemPerBlock < 1024)
            throw std::runtime_error("GPU does not support the 256-thread/1KiB default tile");
        check(cudaStreamCreate(&stream), "Create stream");
        check(cudaEventCreate(&start), "Create start event");
        check(cudaEventCreate(&stop), "Create stop event");
        check(cudaMalloc(&keys, capacity * sizeof(std::uint32_t)), "Allocate keys");
    }
    bool release() {
        bool ok = true;
        auto report = [&](cudaError_t e) {
            if (e != cudaSuccess) { std::cerr << "FAIL: CUDA cleanup: " << cudaGetErrorString(e) << '\n'; ok = false; }
        };
        if (stream) report(cudaStreamSynchronize(stream));
        if (keys) { report(cudaFree(keys)); keys = nullptr; }
        if (start) { report(cudaEventDestroy(start)); start = nullptr; }
        if (stop) { report(cudaEventDestroy(stop)); stop = nullptr; }
        if (stream) { report(cudaStreamDestroy(stream)); stream = nullptr; }
        return ok;
    }
    ~Resources() { release(); }
};
using Sort = void (*)(std::uint32_t*, std::size_t, cudaStream_t);

// Restore before every call, time only kernels, verify every output after timing.
inline float trial(Resources& r, const std::vector<sorting::Key>& original, Sort sort) {
    const auto bytes = original.size() * sizeof(sorting::Key);
    auto output = original;
    if (bytes) check(cudaMemcpyAsync(r.keys, original.data(), bytes,
                                    cudaMemcpyHostToDevice, r.stream), "Restore input");
    check(cudaStreamSynchronize(r.stream), "Finish input restore");
    check(cudaEventRecord(r.start, r.stream), "Record start");
    sort(r.keys, original.size(), r.stream);
    check(cudaEventRecord(r.stop, r.stream), "Record stop");
    check(cudaStreamSynchronize(r.stream), "Wait for sort execution");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, r.start, r.stop), "Read sort time");
    if (bytes) check(cudaMemcpyAsync(output.data(), r.keys, bytes,
                                    cudaMemcpyDeviceToHost, r.stream), "Download output");
    check(cudaStreamSynchronize(r.stream), "Finish download");
    const auto result = sorting::check_sorted(original, output);
    if (!result.passed) throw std::runtime_error(result.message);
    return ms;
}
} // namespace cuda_test
