#include "bitonic_sort.cuh"
#include "bitonic_shared_plan.hpp"
#include "bitonic_internal.cuh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace sorting {
namespace {
__global__ void tile_steps(std::uint32_t* keys, std::size_t n,
                           std::size_t merge_stage, bool initial) {
    extern __shared__ std::uint32_t tile_keys[];
    const unsigned local = threadIdx.x;
    const std::size_t tile = blockDim.x;
    // All threads take the same tile loop and every barrier. No partial tiles.
    for (std::size_t tile_id = blockIdx.x; tile_id < n / tile; tile_id += gridDim.x) {
        const std::size_t i = tile_id * tile + local;
        tile_keys[local] = keys[i];
        __syncthreads();
        const std::size_t last_stage = initial ? tile : merge_stage;
        for (std::size_t stage = initial ? 2 : merge_stage;; stage *= 2) {
            for (std::size_t stride = initial ? stage / 2 : tile / 2;
                 stride > 0; stride /= 2) {
                const auto partner = local ^ stride;
                if (partner > local) {
                    const bool ascending = (i & stage) == 0; // Global index, not local!
                    const auto a = tile_keys[local];
                    const auto b = tile_keys[partner];
                    if (ascending ? a > b : a < b) {
                        tile_keys[local] = b;
                        tile_keys[partner] = a;
                    }
                }
                __syncthreads(); // All pair writes visible to the next step.
            }
            if (stage == last_stage) break;
        }
        keys[i] = tile_keys[local];
        __syncthreads(); // Prevent reuse while another thread is still storing.
    }
}
}

void bitonic_sort_shared(std::uint32_t* keys, std::size_t n, cudaStream_t stream) {
    validate_bitonic_size(n);
    if (n < 2) return;
    if (!keys) throw std::invalid_argument("Null shared Bitonic Sort device pointer");
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
        throw std::invalid_argument("Bitonic Sort allocation size would overflow");
    auto launch_tiles = [&](std::size_t tile, std::size_t stage, bool initial) {
        const auto blocks = static_cast<unsigned>(std::min<std::size_t>(n / tile, 65535));
        tile_steps<<<blocks, static_cast<unsigned>(tile), tile * sizeof(std::uint32_t), stream>>>
            (keys, n, stage, initial);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("Shared Bitonic kernel launch: ") + cudaGetErrorString(error));
    };
    for_each_shared_bitonic_launch(n,
        [&](std::size_t tile) { launch_tiles(tile, tile, true); },
        [&](std::size_t stage, std::size_t stride) {
            detail::launch_global_step(keys, n, stage, stride, stream);
        },
        [&](std::size_t stage, std::size_t tile) { launch_tiles(tile, stage, false); });
}
} // namespace sorting
