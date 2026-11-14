#include "cuda_test_support.cuh"

int main() {
    try {
        sorting::bitonic_sort_shared(nullptr, 0);
        sorting::bitonic_sort_shared(nullptr, 1);
        for (std::size_t n : {3u, 31u, 257u}) {
            bool rejected = false;
            // Non-null placeholder ensures rejection is due to size, not pointer.
            std::uint32_t placeholder = 0;
            try { sorting::bitonic_sort_shared(&placeholder, n); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Unsupported size was not rejected");
        }
        bool null_rejected = false;
        try { sorting::bitonic_sort_shared(nullptr, 2); }
        catch (const std::invalid_argument&) { null_rejected = true; }
        if (!null_rejected) throw std::runtime_error("Null pointer was not rejected");
        cuda_test::Resources r;
        r.initialize(65536);
        const sorting::Distribution distributions[] = {sorting::Distribution::Random,
            sorting::Distribution::Sorted, sorting::Distribution::ReverseSorted,
            sorting::Distribution::AllEqual, sorting::Distribution::DuplicateHeavy,
            sorting::Distribution::BoundaryValues};
        std::size_t cases = 0;
        for (auto distribution : distributions) {
            for (std::size_t n : {0u, 1u, 2u, 4u, 8u, 32u, 128u, 256u, 512u, 1024u, 2048u, 65536u}) {
                const auto input = sorting::generate(n, distribution, 12345);
                try { cuda_test::trial(r, input, sorting::bitonic_sort_shared); }
                catch (const std::exception& error) {
                    throw std::runtime_error("distribution=" + std::to_string(static_cast<int>(distribution))
                        + " n=" + std::to_string(n) + ": " + error.what());
                }
                ++cases;
            }
        }
        if (!r.release()) throw std::runtime_error("Resource cleanup failed");
        std::cout << "PASS: GPU shared Bitonic correctness: " << cases
                  << " cases; no-ops, null pointer, and unsupported sizes verified.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
