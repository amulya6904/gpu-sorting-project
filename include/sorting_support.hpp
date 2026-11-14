#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sorting {
using Key = std::uint32_t;
enum class Distribution { Random, Sorted, ReverseSorted, AllEqual,
                          DuplicateHeavy, BoundaryValues };

// Seed is explicit. Random/Sorted/ReverseSorted share the same initial samples.
// Deterministic distributions ignore the seed. No CUDA or sorting implementation required.
std::vector<Key> generate(std::size_t size, Distribution distribution, Key seed);

struct CheckResult {
    bool passed;
    std::string message;
};

// Neither argument is modified. Compares against std::sort(original copy).
CheckResult check_sorted(const std::vector<Key>& original,
                         const std::vector<Key>& candidate);
} // namespace sorting
