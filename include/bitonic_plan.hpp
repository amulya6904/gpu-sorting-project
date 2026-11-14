#pragma once
#include <cstddef>
#include <stdexcept>

namespace sorting {
inline void validate_bitonic_size(std::size_t n) {
    if (n > 1 && (n & (n - 1)) != 0)
        throw std::invalid_argument("Bitonic Sort requires a power-of-two size (or 0/1)");
}

// Shared host-side schedule for GPU launches and the CPU logic check.
template<class Step>
void for_each_bitonic_step(std::size_t n, Step step) {
    validate_bitonic_size(n);
    if (n < 2) return;
    for (std::size_t stage = 2;; stage *= 2) {
        for (std::size_t stride = stage / 2; stride > 0; stride /= 2)
            step(stage, stride);
        if (stage == n) break; // Avoid overflow after the last stage.
    }
}
} // namespace sorting
