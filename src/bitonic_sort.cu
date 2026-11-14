#include "bitonic_sort.cuh"
#include "bitonic_plan.hpp"
#include "bitonic_internal.cuh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace sorting {
namespace {
__global__ void bitonic_step(std::uint32_t* keys, std::size_t n,
                             std::size_t stage, std::size_t stride) {
    // Grid-stride loop also supports arrays larger than the capped launch grid.
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t partner = i ^ stride;
        if (partner > i && partner < n) {
            const bool ascending = (i & stage) == 0;
            const auto a = keys[i];
            const auto b = keys[partner];
            if (ascending ? a > b : a < b) {
                keys[i] = b;
                keys[partner] = a;
            }
        }
    }
}
}

void detail::launch_global_step(std::uint32_t* device_keys, std::size_t n,
                                std::size_t stage, std::size_t stride, cudaStream_t stream) {
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(
        std::min<std::size_t>((n + threads - 1) / threads, 65535));
    bitonic_step<<<blocks, threads, 0, stream>>>(device_keys, n, stage, stride);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("Bitonic kernel launch: ") + cudaGetErrorString(error));
}

void bitonic_sort(std::uint32_t* device_keys, std::size_t n, cudaStream_t stream) {
    validate_bitonic_size(n); // Reject before any launch, including null pointers.
    if (n < 2) return;
    if (!device_keys) throw std::invalid_argument("Null Bitonic Sort device pointer");
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
        throw std::invalid_argument("Bitonic Sort allocation size would overflow");
    for_each_bitonic_step(n, [&](std::size_t stage, std::size_t stride) {
        detail::launch_global_step(device_keys, n, stage, stride, stream);
    });
}
} // namespace sorting
