#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sorting {
constexpr std::size_t radix_tile_size = 256;
inline void validate_radix_bits(unsigned bits) {
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8)
        throw std::invalid_argument("Radix digit width must be 1, 2, 4, or 8");
}
inline void validate_radix_pass(unsigned bits, unsigned shift) {
    validate_radix_bits(bits);
    if (shift > 32 - bits || shift % bits != 0)
        throw std::invalid_argument("Radix pass shift must be aligned and within 32 bits");
}
inline std::size_t radix_tiles(std::size_t n) {
    return n / radix_tile_size + (n % radix_tile_size != 0);
}
// Number of size_t entries required in EACH histogram/offset allocation.
inline std::size_t radix_workspace_entries(std::size_t n, unsigned bits) {
    validate_radix_bits(bits);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (n > maximum / sizeof(std::uint32_t))
        throw std::invalid_argument("Radix key allocation size would overflow");
    if (n < 2) return 0;
    const auto tiles = radix_tiles(n);
    const std::size_t buckets = std::size_t{1} << bits;
    if (tiles > maximum / buckets / sizeof(std::size_t))
        throw std::invalid_argument("Radix workspace allocation size would overflow");
    return tiles * buckets;
}
template<class Pass>
void for_each_radix_pass(unsigned bits, Pass pass) {
    validate_radix_bits(bits);
    for (unsigned shift = 0; shift < 32; shift += bits) pass(shift);
}
} // namespace sorting
