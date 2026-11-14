#include "bitonic_sort.cuh"
#include "sorting_support.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct Resources {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    std::uint32_t* keys = nullptr;
    bool release() {
        bool ok = true;
        auto report = [&](cudaError_t error) {
            if (error != cudaSuccess) {
                std::cerr << "FAIL: CUDA cleanup: " << cudaGetErrorString(error) << '\n';
                ok = false;
            }
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
}

int main() {
    try {
        // These API paths must not launch kernels or dereference the pointer.
        sorting::bitonic_sort(nullptr, 0);
        sorting::bitonic_sort(nullptr, 1);
        bool rejected = false;
        try { sorting::bitonic_sort(nullptr, 257); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("257 was not rejected");

        int count = 0;
        check(cudaGetDeviceCount(&count), "Query CUDA devices");
        if (count == 0) throw std::runtime_error("No CUDA-capable GPU; GPU verification pending");
        check(cudaSetDevice(0), "Select device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "Read GPU model");
        std::cout << "GPU: " << properties.name << '\n';
        Resources resources;
        check(cudaStreamCreate(&resources.stream), "Create sort stream");
        check(cudaEventCreate(&resources.start), "Create start event");
        check(cudaEventCreate(&resources.stop), "Create stop event");
        check(cudaMalloc(&resources.keys, 65536 * sizeof(std::uint32_t)), "Allocate keys");

        const sorting::Distribution distributions[] = {sorting::Distribution::Random,
            sorting::Distribution::Sorted, sorting::Distribution::ReverseSorted,
            sorting::Distribution::AllEqual, sorting::Distribution::DuplicateHeavy,
            sorting::Distribution::BoundaryValues};
        std::size_t cases = 0;
        float smoke_ms = 0;
        for (auto distribution : distributions) {
            for (std::size_t n : {0u, 1u, 2u, 4u, 8u, 16u, 32u, 256u, 1024u, 65536u}) {
                const auto original = sorting::generate(n, distribution, 12345);
                auto candidate = original;
                const bool timed = n == 65536 && distribution == sorting::Distribution::Random;
                if (n) check(cudaMemcpyAsync(resources.keys, original.data(), n * sizeof(std::uint32_t),
                                             cudaMemcpyHostToDevice, resources.stream), "Upload keys");
                // Explicit wait excludes upload from the smoke interval.
                check(cudaStreamSynchronize(resources.stream), "Finish upload");
                if (timed) check(cudaEventRecord(resources.start, resources.stream), "Record start");
                sorting::bitonic_sort(resources.keys, n, resources.stream);
                if (timed) check(cudaEventRecord(resources.stop, resources.stream), "Record stop");
                check(cudaStreamSynchronize(resources.stream), "Finish sort kernels");
                if (timed) check(cudaEventElapsedTime(&smoke_ms, resources.start, resources.stop), "Read smoke time");
                if (n) check(cudaMemcpyAsync(candidate.data(), resources.keys, n * sizeof(std::uint32_t),
                                             cudaMemcpyDeviceToHost, resources.stream), "Download keys");
                check(cudaStreamSynchronize(resources.stream), "Finish download");
                const auto result = sorting::check_sorted(original, candidate);
                if (!result.passed)
                    throw std::runtime_error("distribution=" + std::to_string(static_cast<int>(distribution))
                        + " n=" + std::to_string(n) + ": " + result.message);
                ++cases;
            }
        }
        if (!resources.release()) throw std::runtime_error("Resource cleanup failed");
        std::cout << "PASS: GPU Bitonic correctness: " << cases
                  << " cases; no-ops and non-power-of-two rejection verified.\n"
                  << "Smoke measurement: random n=65536, full sort kernel sequence = "
                  << smoke_ms << " ms (allocation and copies excluded; not a benchmark).\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
