#include "sorting_support.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using sorting::Distribution;
using sorting::Key;
using sorting::generate;
using sorting::check_sorted;

namespace {
std::size_t checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void expect_rejection(const std::vector<Key>& original,
                      const std::vector<Key>& candidate,
                      const std::string& diagnostic) {
    const auto result = check_sorted(original, candidate);
    require(!result.passed, "Checker accepted invalid output");
    require(result.message.find(diagnostic) != std::string::npos,
            "Wrong diagnostic: " + result.message);
}
}

int main() {
    try {
        const Distribution distributions[] = {Distribution::Random, Distribution::Sorted,
            Distribution::ReverseSorted, Distribution::AllEqual,
            Distribution::DuplicateHeavy, Distribution::BoundaryValues};
        const std::size_t sizes[] = {0, 1, 2, 4, 8, 16, 31, 257, 65536};
        std::size_t cases = 0;
        for (auto distribution : distributions) {
            for (auto size : sizes) {
                ++cases;
                auto input = generate(size, distribution, 12345);
                const auto saved = input;
                require(input.size() == size, "Wrong generated size");
                require(input == generate(size, distribution, 12345), "Seed not repeatable");
                auto sorted = input;
                std::sort(sorted.begin(), sorted.end());
                require(check_sorted(input, sorted).passed, "Rejected correct sort");
                require(input == saved, "Checker modified original");
                if (distribution == Distribution::Sorted || distribution == Distribution::ReverseSorted) {
                    auto reference = generate(size, Distribution::Random, 12345);
                    std::sort(reference.begin(), reference.end());
                    require(sorted == reference, "Ordered distribution changed sample values");
                    if (distribution == Distribution::Sorted)
                        require(std::is_sorted(input.begin(), input.end()), "Not ascending");
                    else
                        require(std::is_sorted(input.begin(), input.end(), std::greater<Key>()), "Not descending");
                }
                if (distribution == Distribution::AllEqual)
                    require(std::all_of(input.begin(), input.end(), [](Key k) { return k == 42; }), "Not all equal");
                if (distribution == Distribution::DuplicateHeavy) {
                    require(std::all_of(input.begin(), input.end(), [](Key k) { return k < 8; }), "Duplicate range");
                    for (Key k = 0; k < 8; ++k)
                        require(static_cast<std::size_t>(std::count(input.begin(), input.end(), k))
                                == size / 8 + (k < size % 8 ? 1 : 0), "Duplicate counts");
                }
                if (distribution == Distribution::BoundaryValues) {
                    const Key maximum = std::numeric_limits<Key>::max();
                    const Key pattern[] = {0, maximum, 1, maximum - 1, Key{1} << 31, (Key{1} << 31) - 1};
                    for (std::size_t i = 0; i < size; ++i)
                        require(input[i] == pattern[i % 6], "Boundary pattern");
                    if (size >= 2) require(input[0] == 0 && input[1] == maximum, "Missing boundaries");
                }
            }
        }
        // Standardized mt19937 seed/output golden sample, independent of library distributions.
        require(generate(3, Distribution::Random, 5489) ==
                std::vector<Key>({3499211612u, 581869302u, 3890346734u}), "Random golden sample");
        for (auto distribution : {Distribution::Random, Distribution::Sorted,
                                 Distribution::ReverseSorted, Distribution::DuplicateHeavy})
            require(generate(257, distribution, 1) != generate(257, distribution, 2), "Seed ignored");

        expect_rejection({3, 1, 2}, {1, 3, 2}, "Mismatch at index 1"); // unsorted
        expect_rejection({3, 1, 2}, {1, 2}, "Size mismatch: expected 3, got 2");
        expect_rejection({1}, {1, 2}, "Size mismatch: expected 1, got 2");
        expect_rejection({1, 2, 3}, {1, 2, 4}, "Mismatch at index 2"); // sorted wrong values
        expect_rejection({1, 1, 2, 3}, {1, 2, 2, 3}, "Mismatch at index 1"); // duplicate counts
        expect_rejection({0}, {std::numeric_limits<Key>::max()}, "Mismatch at index 0");
        expect_rejection({std::numeric_limits<Key>::max(), 0},
                         {std::numeric_limits<Key>::max(), 0}, "Mismatch at index 0");
        require(check_sorted({}, {}).passed, "Empty output rejected");
        require(check_sorted({7}, {7}).passed, "Singleton rejected");
        std::cout << "PASS: " << cases << " distribution/size cases, " << checks
                  << " checks; generator properties and checker rejection tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
