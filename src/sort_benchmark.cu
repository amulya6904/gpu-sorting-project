#include "benchmark_options.hpp"
#include "bitonic_sort.cuh"
#include "radix_sort.cuh"
#include "benchmark_results.hpp"
#include <chrono>
#if ENABLE_CUB_BASELINE
#include <cub/device/device_radix_sort.cuh>
#endif
#include <iomanip>
#include <iostream>

namespace {
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct Resources {
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, start = nullptr, stop = nullptr, end = nullptr;
    std::uint32_t *host_input = nullptr, *host_output = nullptr;
    std::uint32_t *input = nullptr, *output = nullptr;
    sorting::RadixWorkspace radix;
    void* cub_temp = nullptr;
    std::size_t cub_bytes = 0;
    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    bool release(bool configuration_only = false) {
        bool ok = true;
        auto report = [&](cudaError_t error) {
            if (error != cudaSuccess) { std::cerr << "FAIL: cleanup: " << cudaGetErrorString(error) << '\n'; ok = false; }
        };
        if (stream) report(cudaStreamSynchronize(stream));
        if (radix.histogram) { report(cudaFree(radix.histogram)); radix.histogram = nullptr; }
        if (radix.offsets) { report(cudaFree(radix.offsets)); radix.offsets = nullptr; }
        if (cub_temp) { report(cudaFree(cub_temp)); cub_temp = nullptr; }
        if (configuration_only) return ok;
        if (input) { report(cudaFree(input)); input = nullptr; }
        if (output) { report(cudaFree(output)); output = nullptr; }
        if (host_input) { report(cudaFreeHost(host_input)); host_input = nullptr; }
        if (host_output) { report(cudaFreeHost(host_output)); host_output = nullptr; }
        if (begin) { report(cudaEventDestroy(begin)); begin = nullptr; }
        if (end) { report(cudaEventDestroy(end)); end = nullptr; }
        if (start) { report(cudaEventDestroy(start)); start = nullptr; }
        if (stop) { report(cudaEventDestroy(stop)); stop = nullptr; }
        if (stream) { report(cudaStreamDestroy(stream)); stream = nullptr; }
        return ok;
    }
    ~Resources() { release(); }
};

std::uint32_t* enqueue(const std::string& algorithm, std::size_t n, Resources& r) {
    if (algorithm == "bitonic-global") sorting::bitonic_sort(r.input, n, r.stream);
    else if (algorithm == "bitonic-shared") sorting::bitonic_sort_shared(r.input, n, r.stream);
    else if (algorithm.compare(0, 6, "radix-") == 0)
        return sorting::radix_sort(r.input, n, static_cast<unsigned>(algorithm.back() - '0'), r.radix, r.stream);
    else if (algorithm == "cub") {
#if ENABLE_CUB_BASELINE
        if (n == 0) return r.input;
        // Pointer overload: input/output are disjoint; sorted result is output.
        auto available = r.cub_bytes;
        check(cub::DeviceRadixSort::SortKeys(r.cub_temp, available, r.input, r.output,
            static_cast<int>(n), 0, 32, r.stream), "CUB SortKeys execution");
        return r.output;
#else
        throw std::runtime_error("CUB baseline disabled at build time");
#endif
    } else throw std::invalid_argument("Unknown algorithm: " + algorithm);
    return r.input;
}

sorting::TrialTimes trial(const std::string& algorithm, const std::vector<sorting::Key>& original,
             std::vector<sorting::Key>& downloaded, Resources& r) {
    const auto bytes = original.size() * sizeof(sorting::Key);
    check(cudaEventRecord(r.begin, r.stream), "Record H2D start");
    const auto wall_start = std::chrono::steady_clock::now();
    if (bytes) check(cudaMemcpyAsync(r.input, r.host_input, bytes, cudaMemcpyHostToDevice, r.stream), "Restore original input");
    check(cudaEventRecord(r.start, r.stream), "Record sort start");
    auto* actual_output = enqueue(algorithm, original.size(), r);
    check(cudaGetLastError(), "Sort launch status");
    check(cudaEventRecord(r.stop, r.stream), "Record sort stop");
    if (bytes) check(cudaMemcpyAsync(r.host_output, actual_output, bytes, cudaMemcpyDeviceToHost, r.stream), "Download actual output buffer");
    check(cudaEventRecord(r.end, r.stream), "Record D2H stop");
    check(cudaStreamSynchronize(r.stream), "Finish result download");
    const auto wall_stop = std::chrono::steady_clock::now();
    float h2d = 0, sort = 0, d2h = 0;
    check(cudaEventElapsedTime(&h2d, r.begin, r.start), "Read H2D time");
    check(cudaEventElapsedTime(&sort, r.start, r.stop), "Read sort time");
    check(cudaEventElapsedTime(&d2h, r.stop, r.end), "Read D2H time");
    if (bytes) std::copy(r.host_output, r.host_output + original.size(), downloaded.begin());
    const auto result = sorting::check_sorted(original, downloaded);
    if (!result.passed) throw std::runtime_error(algorithm + ": " + result.message);
    return {h2d, sort, d2h, std::chrono::duration<double, std::milli>(wall_stop - wall_start).count()};
}


}

int main(int argc, char** argv) {
    try {
        const auto options = sorting::parse_benchmark_options(std::vector<std::string>(argv + 1, argv + argc));
        if (options.help) {
            std::cout << "sort_benchmark --algorithm all|NAME[,NAME...] --size N --distribution NAME --seed N --warmups N --trials N\n"
                      << "Algorithms: bitonic-global,bitonic-shared,radix-1,radix-2,radix-4,radix-8,cub\n"
                      << "Distributions: random,sorted,reverse-sorted,all-equal,duplicate-heavy,boundary-value\n"
                      << "Defaults: all,1024,random,12345,3,10. --output-prefix PATH exports CSVs. Size range: 0..INT_MAX.\n";
            return 0;
        }
        if (std::string(SORT_BUILD_CONFIGURATION) != "Release")
            throw std::runtime_error("Measurements require a Release build");
        std::cout << "Build configuration: " << SORT_BUILD_CONFIGURATION
                  << "\nCUDA Toolkit CUDART_VERSION: " << CUDART_VERSION << '\n';
        // Generated exactly once; all algorithms/trials restore this same original.
        const auto original = sorting::generate(options.size, options.distribution, options.seed);
        auto downloaded = original;
        Resources r;
        int count = 0;
        check(cudaGetDeviceCount(&count), "Query CUDA GPUs");
        if (!count) throw std::runtime_error("No CUDA GPU; smoke results and timings pending");
        check(cudaSetDevice(0), "Select GPU");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "Read GPU model");
        int driver_api = 0, runtime_version = 0;
        check(cudaDriverGetVersion(&driver_api), "Read CUDA driver API version");
        check(cudaRuntimeGetVersion(&runtime_version), "Read CUDA runtime version");
        std::cout << "CUDA driver API version: " << driver_api << "\nCUDA runtime version: " << runtime_version << '\n';
        check(cudaStreamCreate(&r.stream), "Create sorting stream");
        check(cudaEventCreate(&r.start), "Create start event");
        check(cudaEventCreate(&r.stop), "Create stop event");
        check(cudaEventCreate(&r.begin), "Create H2D start event");
        check(cudaEventCreate(&r.end), "Create D2H stop event");
        const auto bytes = std::max<std::size_t>(options.size, 1) * sizeof(sorting::Key);
        check(cudaMalloc(&r.input, bytes), "Allocate input keys");
        check(cudaMalloc(&r.output, bytes), "Allocate disjoint output/alternate keys");
        check(cudaMallocHost(&r.host_input, bytes), "Allocate pinned input");
        check(cudaMallocHost(&r.host_output, bytes), "Allocate pinned output");
        std::copy(original.begin(), original.end(), r.host_input);
        r.radix.alternate = r.output;
        r.radix.alternate_capacity = options.size;
        std::cout << "GPU: " << properties.name << "\nCase: n=" << options.size
                  << " distribution=" << options.distribution_name << " seed=" << options.seed
                  << " warmups=" << options.warmups << " trials=" << options.trials
                  << "\nSort-only CUDA-event milliseconds; each warmup/trial fully verified.\n";
        unsigned passed = 0, unsupported = 0, unavailable = 0;
        bool failed = false;
        std::vector<sorting::BenchmarkResult> results;
        for (const auto& algorithm : options.algorithms) {
            results.push_back({algorithm, "PASS", "", {}});
            auto& result = results.back();
            if (failed) { result.status = "SKIPPED"; result.reason = "Earlier configuration failed; CUDA state not reused"; continue; }
            if (!sorting::benchmark_supported(algorithm, options.size)) {
                result.status = "SKIPPED"; result.reason = "Bitonic requires power-of-two n (or 0/1)";
                std::cout << "UNSUPPORTED: " << algorithm << ": requires power-of-two n (or 0/1)\n";
                ++unsupported;
                continue;
            }
#if !ENABLE_CUB_BASELINE
            if (algorithm == "cub") {
                result.status = "SKIPPED"; result.reason = "CUB disabled at build time";
                std::cout << "UNAVAILABLE: cub: build with ENABLE_CUB_BASELINE=ON\n";
                ++unavailable;
                continue;
            }
#endif
            try {
            if (algorithm.compare(0, 6, "radix-") == 0) {
                r.radix.entries = sorting::radix_workspace_entries(options.size, static_cast<unsigned>(algorithm.back() - '0'));
                if (r.radix.entries) {
                    check(cudaMalloc(&r.radix.histogram, r.radix.entries * sizeof(std::size_t)), "Allocate radix histogram");
                    check(cudaMalloc(&r.radix.offsets, r.radix.entries * sizeof(std::size_t)), "Allocate radix offsets");
                }
            }
#if ENABLE_CUB_BASELINE
            if (algorithm == "cub" && options.size) {
                r.cub_bytes = 0;
                check(cub::DeviceRadixSort::SortKeys(nullptr, r.cub_bytes, r.input, r.output,
                    static_cast<int>(options.size), 0, 32, r.stream), "Query CUB temporary storage");
                // A non-null pointer distinguishes execution from query, even if zero bytes requested.
                check(cudaMalloc(&r.cub_temp, std::max<std::size_t>(r.cub_bytes, 1)), "Allocate CUB temporary storage");
                std::cout << "CUB temporary storage: " << r.cub_bytes << " bytes (reused)\n";
            }
#endif
            for (unsigned i = 0; i < options.warmups; ++i) trial(algorithm, original, downloaded, r);
            for (unsigned i = 0; i < options.trials; ++i) result.trials.push_back(trial(algorithm, original, downloaded, r));
            const auto median = sorting::medians(result.trials).sort;
            if (!r.release(true)) throw std::runtime_error("Configuration cleanup failed");
            std::cout << "PASS: " << algorithm << " median_sort_ms=" << std::fixed << std::setprecision(6) << median << '\n';
            ++passed;
            } catch (const std::exception& error) {
                result.status = "FAIL"; result.reason = error.what(); failed = true;
                std::cerr << "FAIL: " << algorithm << ": " << error.what() << '\n';
            }
        }
        sorting::write_benchmark_csv(options, properties.name, SORT_BUILD_CONFIGURATION, results);
        if (!r.release()) throw std::runtime_error("Runner cleanup failed");
        std::cout << "Summary: passed=" << passed << " unsupported=" << unsupported << " unavailable=" << unavailable << '\n';
        return failed || unavailable ? 1 : passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
