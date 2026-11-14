#include "sorting_support.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

namespace sorting {
std::vector<Key> generate(std::size_t size, Distribution distribution, Key seed) {
    std::vector<Key> keys(size);
    std::mt19937 engine(seed);
    // Use engine outputs directly: reproducible across standard libraries.
    switch (distribution) {
    case Distribution::Random:
    case Distribution::Sorted:
    case Distribution::ReverseSorted:
        for (auto& key : keys) key = static_cast<Key>(engine());
        if (distribution == Distribution::Sorted)
            std::sort(keys.begin(), keys.end());
        if (distribution == Distribution::ReverseSorted)
            std::sort(keys.begin(), keys.end(), std::greater<Key>());
        break;
    case Distribution::AllEqual:
        std::fill(keys.begin(), keys.end(), Key{42});
        break;
    case Distribution::DuplicateHeavy:
        // Only eight possible values; each group of eight includes every value.
        for (std::size_t i = 0; i < size; ++i) keys[i] = static_cast<Key>(i % 8);
        // Explicit Fisher-Yates mapping avoids library-specific shuffle behavior.
        for (std::size_t count = size; count > 1; --count) {
            const auto j = static_cast<std::size_t>(engine()) % count;
            std::swap(keys[count - 1], keys[j]);
        }
        break;
    case Distribution::BoundaryValues: {
        const Key maximum = std::numeric_limits<Key>::max();
        const Key values[] = {0, maximum, 1, maximum - 1,
                              Key{1} << 31, (Key{1} << 31) - 1};
        for (std::size_t i = 0; i < size; ++i) keys[i] = values[i % 6];
        break;
    }
    default:
        throw std::invalid_argument("Unknown input distribution");
    }
    return keys;
}

CheckResult check_sorted(const std::vector<Key>& original,
                         const std::vector<Key>& candidate) {
    if (original.size() != candidate.size()) {
        return {false, "Size mismatch: expected " + std::to_string(original.size())
                     + ", got " + std::to_string(candidate.size())};
    }
    auto reference = original;
    std::sort(reference.begin(), reference.end());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (reference[i] != candidate[i]) {
            return {false, "Mismatch at index " + std::to_string(i)
                         + ": expected " + std::to_string(reference[i])
                         + ", got " + std::to_string(candidate[i])};
        }
    }
    return {true, "PASS: all " + std::to_string(reference.size()) + " keys match"};
}
} // namespace sorting
