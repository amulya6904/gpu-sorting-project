#include "radix_plan.hpp"
#include "radix_test_cases.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
// CPU model of the GPU histogram/offset/rank equations; no CUDA executed.
std::vector<sorting::Key> pass(const std::vector<sorting::Key>& input,
                               unsigned bits, unsigned shift) {
    sorting::validate_radix_pass(bits, shift);
    if (input.size() < 2) return input;
    const auto n = input.size(), tiles = sorting::radix_tiles(n);
    const unsigned buckets = 1u << bits, mask = buckets - 1;
    std::vector<std::size_t> histogram(sorting::radix_workspace_entries(n, bits), 0);
    for (std::size_t i = 0; i < n; ++i)
        ++histogram[static_cast<std::size_t>((input[i] >> shift) & mask) * tiles + i / 256];
    std::vector<std::size_t> offsets(histogram.size());
    std::size_t base = 0;
    for (unsigned digit = 0; digit < buckets; ++digit) {
        std::size_t earlier_tiles = 0;
        for (std::size_t tile = 0; tile < tiles; ++tile) {
            const auto entry = static_cast<std::size_t>(digit) * tiles + tile;
            offsets[entry] = base + earlier_tiles;
            earlier_tiles += histogram[entry];
        }
        base += earlier_tiles;
    }
    require(base == n, "Histogram lost keys");
    std::vector<sorting::Key> output(n);
    std::vector<unsigned> visits(n, 0);
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        std::vector<std::size_t> ranks(buckets, 0);
        for (std::size_t i = tile * 256; i < std::min(n, (tile + 1) * 256); ++i) {
            const auto digit = (input[i] >> shift) & mask;
            // Serial input traversal computes count of earlier matching local keys.
            const auto destination = offsets[static_cast<std::size_t>(digit) * tiles + tile] + ranks[digit]++;
            require(destination < n, "Destination out of bounds");
            ++visits[destination];
            output[destination] = input[i];
        }
    }
    for (auto count : visits) require(count == 1, "Scatter hole or duplicate destination");
    return output;
}
}

int main() {
    try {
        const sorting::Distribution distributions[] = {sorting::Distribution::Random,
            sorting::Distribution::Sorted, sorting::Distribution::ReverseSorted,
            sorting::Distribution::AllEqual, sorting::Distribution::DuplicateHeavy,
            sorting::Distribution::BoundaryValues};
        std::size_t cases = 0, stable_cases = 0;
        for (unsigned bits : {1u, 2u, 4u, 8u}) {
            for (auto distribution : distributions) {
                for (std::size_t n : {0u, 1u, 2u, 31u, 255u, 256u, 257u, 511u, 513u, 65536u}) {
                    const auto input = sorting::generate(n, distribution, 12345);
                    auto output = input;
                    unsigned passes = 0;
                    sorting::for_each_radix_pass(bits, [&](unsigned shift) {
                        output = pass(output, bits, shift);
                        ++passes;
                    });
                    require(passes == 32 / bits && passes % 2 == 0, "Wrong pass count/buffer parity");
                    const auto result = sorting::check_sorted(input, output);
                    if (!result.passed) throw std::runtime_error(result.message);
                    ++cases;
                }
            }
            for (unsigned shift : {0u, 32u - bits}) {
                for (bool same : {false, true}) {
                    const auto input = radix_test::stable_input(bits, shift, same);
                    require(pass(input, bits, shift) == radix_test::stable_reference(input, bits, shift),
                            "Single-pass stability failed");
                    ++stable_cases;
                }
            }
        }
        for (unsigned bits : {0u, 3u, 7u, 16u, 32u}) {
            bool invoked = false, rejected = false;
            try { sorting::for_each_radix_pass(bits, [&](unsigned) { invoked = true; }); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && !invoked, "Invalid width was not rejected before passes");
        }
        for (unsigned shift : {1u, 29u, 32u}) {
            bool rejected = false;
            try { sorting::validate_radix_pass(4, shift); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid pass shift accepted");
        }
        bool overflow_rejected = false;
        try { sorting::radix_workspace_entries(std::numeric_limits<std::size_t>::max(), 8); }
        catch (const std::invalid_argument&) { overflow_rejected = true; }
        require(overflow_rejected, "Overflow size accepted");
        require(sorting::radix_workspace_entries(0, 8) == 0 && sorting::radix_workspace_entries(1, 8) == 0,
                "No-op workspace requirements");
        require(sorting::radix_workspace_entries(257, 8) == 512, "Partial tile workspace requirements");
        std::cout << "PASS: CPU radix logic: " << cases << " full-sort cases, " << stable_cases
                  << " stable-pass cases; scatter destinations, widths, shifts, workspace, and pass counts verified. GPU NOT tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
