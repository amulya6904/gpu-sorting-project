#pragma once
#include "radix_plan.hpp"
#include <cuda_runtime.h>

namespace sorting {
struct RadixWorkspace {
    std::uint32_t* alternate = nullptr;
    std::size_t alternate_capacity = 0;
    std::size_t* histogram = nullptr;
    std::size_t* offsets = nullptr;
    std::size_t entries = 0; // Capacity in size_t entries of EACH array above.
};

// All allocations/transfers belong to the caller. For n>1 all device arrays
// must be disjoint and sized as declared. Workspace can be reused on the same
// ordered stream; concurrent sorts need separate storage. Does not allocate/wait.
// Returns the final device pointer. All supported full sorts have an even pass
// count, so the result is in keys; alternate holds the preceding pass's output.
// n=0/1 returns keys without accessing memory/workspace. Width is still validated.
// Launch errors throw; caller must check cudaStreamSynchronize for execution errors.
std::uint32_t* radix_sort(std::uint32_t* keys, std::size_t n, unsigned bits,
                          const RadixWorkspace& workspace, cudaStream_t stream = nullptr);

// Exposed for stable-scatter testing/reuse: one stable digit partition, out of place.
// Output has n keys; input/output/histogram/offsets must be disjoint. Requires
// radix_workspace_entries(n,bits) histogram/offset entries. n=0/1 are no-ops;
// singleton callers retain the original input (no output copy). Returns input
// for n<2, otherwise output. shift must be aligned to bits and within the key.
const std::uint32_t* radix_pass(const std::uint32_t* input, std::uint32_t* output,
                               std::size_t n, unsigned bits, unsigned shift,
                               std::size_t* histogram, std::size_t* offsets,
                               std::size_t entries, cudaStream_t stream = nullptr);
} // namespace sorting
