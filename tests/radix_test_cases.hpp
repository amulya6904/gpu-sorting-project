#pragma once
#include "sorting_support.hpp"
#include <algorithm>
#include <cstdint>

namespace radix_test {
inline std::vector<sorting::Key> stable_input(unsigned bits, unsigned shift, bool same_digit) {
    std::vector<sorting::Key> input(769); // Three full tiles and one partial tile.
    const std::uint64_t low_mask = (std::uint64_t{1} << shift) - 1;
    const unsigned digit_mask = (1u << bits) - 1;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const std::uint64_t payload = input.size() - 1 - i; // Unique and deliberately descending.
        const unsigned digit = same_digit ? digit_mask : static_cast<unsigned>(i * 7 + 3) & digit_mask;
        input[i] = static_cast<sorting::Key>((payload & low_mask)
            | ((payload >> shift) << (shift + bits)) | (std::uint64_t{digit} << shift));
    }
    return input;
}
inline std::vector<sorting::Key> stable_reference(std::vector<sorting::Key> input,
                                                 unsigned bits, unsigned shift) {
    const unsigned mask = (1u << bits) - 1;
    std::stable_sort(input.begin(), input.end(), [=](sorting::Key a, sorting::Key b) {
        return ((a >> shift) & mask) < ((b >> shift) & mask);
    });
    return input;
}
} // namespace radix_test
