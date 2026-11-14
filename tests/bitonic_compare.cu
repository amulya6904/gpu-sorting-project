#include "cuda_test_support.cuh"
#include <algorithm>
#include <iomanip>

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int main() {
    try {
        cuda_test::Resources r;
        r.initialize(65536);
        std::cout << "Sort-only CUDA events; seed=12345; 2 warmups/version, 7 trials/version.\n"
                  << "n,global_median_ms,shared_median_ms,speedup_global_over_shared\n";
        for (std::size_t n : {256u, 4096u, 65536u}) {
            const auto input = sorting::generate(n, sorting::Distribution::Random, 12345);
            for (int warmup = 0; warmup < 2; ++warmup) {
                cuda_test::trial(r, input, sorting::bitonic_sort);
                cuda_test::trial(r, input, sorting::bitonic_sort_shared);
            }
            std::vector<float> global_times, shared_times;
            for (int trial = 0; trial < 7; ++trial) {
                // Alternate order to reduce systematic first/second-run bias.
                if (trial % 2 == 0) {
                    global_times.push_back(cuda_test::trial(r, input, sorting::bitonic_sort));
                    shared_times.push_back(cuda_test::trial(r, input, sorting::bitonic_sort_shared));
                } else {
                    shared_times.push_back(cuda_test::trial(r, input, sorting::bitonic_sort_shared));
                    global_times.push_back(cuda_test::trial(r, input, sorting::bitonic_sort));
                }
            }
            const float global_ms = median(global_times), shared_ms = median(shared_times);
            std::cout << std::fixed << std::setprecision(6) << n << ',' << global_ms << ',' << shared_ms << ',';
            if (shared_ms > 0) std::cout << global_ms / shared_ms;
            else std::cout << "N/A (shared time below event resolution)";
            std::cout << '\n';
        }
        if (!r.release()) throw std::runtime_error("Resource cleanup failed");
        std::cout << "PASS: all warmup/trial outputs matched std::sort. Small comparison only.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
