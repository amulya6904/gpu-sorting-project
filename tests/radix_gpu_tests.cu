#include "radix_sort.cuh"
#include "cuda_test_support.cuh"
#include "radix_test_cases.hpp"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct WorkspaceOwner {
    sorting::RadixWorkspace workspace;
    bool release() {
        bool ok = true;
        auto report = [&](cudaError_t error) {
            if (error != cudaSuccess) {
                std::cerr << "FAIL: Radix workspace cleanup: " << cudaGetErrorString(error) << '\n'; ok = false;
            }
        };
        if (workspace.alternate) { report(cudaFree(workspace.alternate)); workspace.alternate = nullptr; }
        if (workspace.histogram) { report(cudaFree(workspace.histogram)); workspace.histogram = nullptr; }
        if (workspace.offsets) { report(cudaFree(workspace.offsets)); workspace.offsets = nullptr; }
        return ok;
    }
    ~WorkspaceOwner() { release(); }
};
}
int main() {
    try {
        sorting::RadixWorkspace empty;
        for (unsigned bits : {1u, 2u, 4u, 8u}) {
            require(sorting::radix_sort(nullptr, 0, bits, empty) == nullptr, "Empty not a no-op");
            require(sorting::radix_sort(nullptr, 1, bits, empty) == nullptr, "Singleton not a no-op");
        }
        for (unsigned bits : {0u, 3u, 7u, 16u, 32u}) {
            bool rejected = false;
            try { sorting::radix_sort(nullptr, 0, bits, empty); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Unsupported width accepted");
        }
        // Reverse destruction order drains the stream before freeing workspace.
        WorkspaceOwner owner;
        cuda_test::Resources r;
        r.initialize(65536);
        auto& w = owner.workspace;
        w.alternate_capacity = 65536;
        w.entries = sorting::radix_workspace_entries(65536, 8);
        cuda_test::check(cudaMalloc(&w.alternate, 65536 * sizeof(std::uint32_t)), "Allocate radix alternate");
        cuda_test::check(cudaMalloc(&w.histogram, w.entries * sizeof(std::size_t)), "Allocate histogram");
        cuda_test::check(cudaMalloc(&w.offsets, w.entries * sizeof(std::size_t)), "Allocate offsets");
        std::size_t cases = 0, stable_cases = 0;
        try {
            const sorting::Distribution distributions[] = {sorting::Distribution::Random,
                sorting::Distribution::Sorted, sorting::Distribution::ReverseSorted,
                sorting::Distribution::AllEqual, sorting::Distribution::DuplicateHeavy,
                sorting::Distribution::BoundaryValues};
            for (unsigned bits : {1u, 2u, 4u, 8u}) {
                for (auto distribution : distributions) {
                    for (std::size_t n : {0u, 1u, 2u, 31u, 255u, 256u, 257u, 511u, 513u, 65536u}) {
                        const auto input = sorting::generate(n, distribution, 12345);
                        auto output = input;
                        const auto bytes = n * sizeof(std::uint32_t);
                        if (n) cuda_test::check(cudaMemcpyAsync(r.keys, input.data(), bytes,
                            cudaMemcpyHostToDevice, r.stream), "Upload radix input");
                        cuda_test::check(cudaStreamSynchronize(r.stream), "Finish radix upload");
                        auto* final_keys = sorting::radix_sort(r.keys, n, bits, w, r.stream);
                        require(final_keys == r.keys, "Wrong full-sort final buffer");
                        cuda_test::check(cudaStreamSynchronize(r.stream), "Wait for radix execution");
                        if (n) cuda_test::check(cudaMemcpyAsync(output.data(), final_keys, bytes,
                            cudaMemcpyDeviceToHost, r.stream), "Download radix result");
                        cuda_test::check(cudaStreamSynchronize(r.stream), "Wait for radix download");
                        const auto result = sorting::check_sorted(input, output);
                        if (!result.passed) throw std::runtime_error("bits=" + std::to_string(bits)
                            + " distribution=" + std::to_string(static_cast<int>(distribution))
                            + " n=" + std::to_string(n) + ": " + result.message);
                        ++cases;
                    }
                }
                for (unsigned shift : {0u, 32u - bits}) {
                    for (bool same : {false, true}) {
                        const auto input = radix_test::stable_input(bits, shift, same);
                        const auto reference = radix_test::stable_reference(input, bits, shift);
                        auto output = input;
                        const auto bytes = input.size() * sizeof(std::uint32_t);
                        cuda_test::check(cudaMemcpyAsync(r.keys, input.data(), bytes,
                            cudaMemcpyHostToDevice, r.stream), "Upload stable-pass input");
                        cuda_test::check(cudaStreamSynchronize(r.stream), "Finish stable-pass upload");
                        const auto* final_keys = sorting::radix_pass(r.keys, w.alternate, input.size(),
                            bits, shift, w.histogram, w.offsets, w.entries, r.stream);
                        require(final_keys == w.alternate, "Wrong single-pass output buffer");
                        cuda_test::check(cudaStreamSynchronize(r.stream), "Wait for stable scatter");
                        cuda_test::check(cudaMemcpyAsync(output.data(), final_keys, bytes,
                            cudaMemcpyDeviceToHost, r.stream), "Download stable-pass output");
                        cuda_test::check(cudaStreamSynchronize(r.stream), "Finish stable-pass download");
                        require(output == reference, "Stable scatter differs from CPU stable sort by digit");
                        ++stable_cases;
                    }
                }
            }
            bool rejected = false;
            auto too_small = w;
            too_small.entries = 0;
            try { sorting::radix_sort(r.keys, 257, 8, too_small, r.stream); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Undersized workspace accepted");
        } catch (...) {
            // Drain queued work before either allocation owner is destroyed.
            cuda_test::check(cudaStreamSynchronize(r.stream), "Drain failed radix test");
            throw;
        }
        cuda_test::check(cudaStreamSynchronize(r.stream), "Finish radix tests");
        const bool workspace_ok = owner.release();
        const bool resources_ok = r.release();
        require(workspace_ok && resources_ok, "CUDA cleanup failed");
        std::cout << "PASS: GPU radix correctness: " << cases << " full-sort cases across all four widths; "
                  << stable_cases << " stable-scatter cases; no-ops, final buffers, and rejection verified.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
