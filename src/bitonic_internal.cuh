#pragma once
#include "bitonic_sort.cuh"

namespace sorting {
namespace detail {
// Shared by the two implementations for cross-tile compare steps.
void launch_global_step(std::uint32_t* keys, std::size_t n,
                        std::size_t stage, std::size_t stride, cudaStream_t stream);
}
}
