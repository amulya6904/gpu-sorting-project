#include "bitonic_plan.hpp"
#include "sorting_support.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        const sorting::Distribution distributions[] = {sorting::Distribution::Random,
            sorting::Distribution::Sorted, sorting::Distribution::ReverseSorted,
            sorting::Distribution::AllEqual, sorting::Distribution::DuplicateHeavy,
            sorting::Distribution::BoundaryValues};
        std::size_t cases = 0;
        for (auto distribution : distributions) {
            for (std::size_t n : {0u, 1u, 2u, 4u, 8u, 16u, 32u, 256u, 1024u, 65536u}) {
                const auto original = sorting::generate(n, distribution, 12345);
                auto candidate = original;
                std::size_t steps = 0;
                sorting::for_each_bitonic_step(n, [&](std::size_t stage, std::size_t stride) {
                    ++steps;
                    std::vector<unsigned> visits(n, 0);
                    for (std::size_t i = 0; i < n; ++i) {
                        const auto partner = i ^ stride;
                        if (partner >= n || (partner ^ stride) != i)
                            throw std::runtime_error("Invalid partner");
                        if (partner > i) {
                            ++visits[i]; ++visits[partner];
                            const bool ascending = (i & stage) == 0;
                            if (ascending ? candidate[i] > candidate[partner]
                                          : candidate[i] < candidate[partner])
                                std::swap(candidate[i], candidate[partner]);
                        }
                    }
                    for (auto count : visits)
                        if (count != 1) throw std::runtime_error("Pair ownership not unique");
                });
                unsigned levels = 0;
                for (auto value = n; value > 1; value /= 2) ++levels;
                if (steps != levels * (levels + 1) / 2)
                    throw std::runtime_error("Wrong step count");
                const auto result = sorting::check_sorted(original, candidate);
                if (!result.passed) throw std::runtime_error(result.message);
                ++cases;
            }
        }
        for (std::size_t n : {3u, 31u, 257u}) {
            bool rejected = false;
            bool invoked = false;
            try { sorting::for_each_bitonic_step(n, [&](std::size_t, std::size_t) { invoked = true; }); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected || invoked) throw std::runtime_error("Invalid size was not rejected before steps");
        }
        std::cout << "PASS: CPU Bitonic logic check: " << cases
                  << " cases; unique pairs, step counts, full results, and invalid sizes verified. GPU NOT tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
