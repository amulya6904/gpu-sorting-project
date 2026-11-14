#include "radix_sort.cuh"
#include <algorithm>
#include <string>
#include <utility>

namespace sorting {
namespace {
void check_launch(const char* operation) {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__global__ void histogram_step(const std::uint32_t* input, std::size_t n,
                                unsigned bits, unsigned shift, std::size_t* histogram) {
    __shared__ unsigned counts[256];
    const unsigned local = threadIdx.x, buckets = 1u << bits;
    const auto tiles = n / 256 + (n % 256 != 0);
    for (std::size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        if (local < buckets) counts[local] = 0;
        __syncthreads();
        const auto i = tile * 256 + local;
        if (i < n) atomicAdd(&counts[(input[i] >> shift) & (buckets - 1)], 1u);
        __syncthreads();
        if (local < buckets) histogram[static_cast<std::size_t>(local) * tiles + tile] = counts[local];
        __syncthreads(); // Protect counts before the next tile reuses them.
    }
}

// Correctness-first, self-implemented GPU scan. One thread per digit scans all
// tiles in input order, then computes its exclusive base from lower digit totals.
// No library scan or sort primitives. This scan is intentionally not optimized.
__global__ void prefix_step(const std::size_t* histogram, std::size_t* offsets,
                             std::size_t tiles, unsigned buckets) {
    __shared__ std::size_t totals[256];
    const unsigned digit = threadIdx.x;
    std::size_t sum = 0;
    if (digit < buckets) {
        for (std::size_t tile = 0; tile < tiles; ++tile) {
            const auto entry = static_cast<std::size_t>(digit) * tiles + tile;
            offsets[entry] = sum;
            sum += histogram[entry];
        }
    }
    totals[digit] = sum;
    __syncthreads();
    if (digit < buckets) {
        std::size_t base = 0;
        for (unsigned earlier = 0; earlier < digit; ++earlier) base += totals[earlier];
        for (std::size_t tile = 0; tile < tiles; ++tile)
            offsets[static_cast<std::size_t>(digit) * tiles + tile] += base;
    }
}

__global__ void scatter_step(const std::uint32_t* input, std::uint32_t* output,
                              std::size_t n, unsigned bits, unsigned shift,
                              const std::size_t* offsets) {
    __shared__ unsigned digits[256];
    const unsigned local = threadIdx.x, mask = (1u << bits) - 1;
    const auto tiles = n / 256 + (n % 256 != 0);
    for (std::size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const auto i = tile * 256 + local;
        std::uint32_t key = 0;
        if (i < n) key = input[i];
        const unsigned digit = (key >> shift) & mask;
        digits[local] = i < n ? digit : 256; // Sentinel outside every supported digit range.
        __syncthreads();
        if (i < n) {
            std::size_t rank = 0;
            // Rank by original local index, never by thread arrival/atomic order.
            for (unsigned earlier = 0; earlier < local; ++earlier)
                rank += digits[earlier] == digit;
            const auto destination = offsets[static_cast<std::size_t>(digit) * tiles + tile] + rank;
            output[destination] = key;
        }
        __syncthreads(); // All readers finish before shared digits are reused.
    }
}
}

const std::uint32_t* radix_pass(const std::uint32_t* input, std::uint32_t* output,
                               std::size_t n, unsigned bits, unsigned shift,
                               std::size_t* histogram, std::size_t* offsets,
                               std::size_t entries, cudaStream_t stream) {
    validate_radix_pass(bits, shift);
    const auto required = radix_workspace_entries(n, bits);
    if (n < 2) return input;
    if (!input || !output || input == output || !histogram || !offsets || histogram == offsets)
        throw std::invalid_argument("Radix pass requires distinct non-null device buffers");
    if (entries < required) throw std::invalid_argument("Radix histogram/offset workspace too small");
    const auto tiles = radix_tiles(n);
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(tiles, 65535));
    histogram_step<<<blocks, 256, 0, stream>>>(input, n, bits, shift, histogram);
    check_launch("Radix histogram launch");
    prefix_step<<<1, 256, 0, stream>>>(histogram, offsets, tiles, 1u << bits);
    check_launch("Radix prefix launch");
    scatter_step<<<blocks, 256, 0, stream>>>(input, output, n, bits, shift, offsets);
    check_launch("Radix scatter launch");
    return output;
}

std::uint32_t* radix_sort(std::uint32_t* keys, std::size_t n, unsigned bits,
                          const RadixWorkspace& workspace, cudaStream_t stream) {
    const auto required = radix_workspace_entries(n, bits); // Width checked even for no-ops.
    if (n < 2) return keys;
    if (!keys || !workspace.alternate || keys == workspace.alternate ||
        !workspace.histogram || !workspace.offsets || workspace.histogram == workspace.offsets)
        throw std::invalid_argument("Radix Sort requires distinct non-null device buffers");
    if (workspace.alternate_capacity < n || workspace.entries < required)
        throw std::invalid_argument("Radix Sort workspace too small");
    auto input = keys;
    auto output = workspace.alternate;
    for_each_radix_pass(bits, [&](unsigned shift) {
        radix_pass(input, output, n, bits, shift, workspace.histogram,
                   workspace.offsets, workspace.entries, stream);
        std::swap(input, output);
    });
    return input;
}
} // namespace sorting
