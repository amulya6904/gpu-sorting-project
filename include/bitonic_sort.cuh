#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace sorting {
// Enqueues an in-place ascending sort on stream; no allocation, copies, or wait.
// n=0/1 is a no-op (nullptr allowed); otherwise device_keys must hold n keys.
// Throws on invalid size/pointer or CUDA launch error. Caller must wait on stream
// to surface asynchronous execution errors and keep the allocation alive.
void bitonic_sort(std::uint32_t* device_keys, std::size_t n,
                  cudaStream_t stream = nullptr);

// Same input/stream/error contract; hybrid global/shared-memory implementation.
// Default tile is 256 keys (256 threads, 1024 shared bytes), smaller for n<256.
void bitonic_sort_shared(std::uint32_t* device_keys, std::size_t n,
                         cudaStream_t stream = nullptr);
} // namespace sorting
