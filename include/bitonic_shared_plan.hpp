#pragma once
#include "bitonic_plan.hpp"
#include <algorithm>

namespace sorting {
constexpr std::size_t bitonic_default_tile = 256;

// Same schedule drives the CUDA launches and CPU simulation. Every tile is full.
template<class Initial, class Global, class Finish>
void for_each_shared_bitonic_launch(std::size_t n, Initial initial,
                                    Global global, Finish finish) {
    validate_bitonic_size(n);
    if (n < 2) return;
    const auto tile = std::min(n, bitonic_default_tile);
    initial(tile);
    if (tile == n) return;
    for (std::size_t stage = tile * 2;; stage *= 2) {
        for (std::size_t stride = stage / 2; stride >= tile; stride /= 2)
            global(stage, stride);
        finish(stage, tile);
        if (stage == n) break;
    }
}
} // namespace sorting
