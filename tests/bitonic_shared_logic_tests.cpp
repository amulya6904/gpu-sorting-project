#include "bitonic_shared_plan.hpp"
#include "sorting_support.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}
int main() {
    try {
        const sorting::Distribution distributions[] = {sorting::Distribution::Random,
            sorting::Distribution::Sorted, sorting::Distribution::ReverseSorted,
            sorting::Distribution::AllEqual, sorting::Distribution::DuplicateHeavy,
            sorting::Distribution::BoundaryValues};
        std::size_t cases = 0;
        for (auto distribution : distributions) {
            for (std::size_t n : {0u, 1u, 2u, 4u, 8u, 32u, 128u, 256u, 512u, 1024u, 2048u, 65536u}) {
                const auto input = sorting::generate(n, distribution, 12345);
                auto output = input;
                const auto tile = std::min(n, sorting::bitonic_default_tile);
                std::vector<std::pair<std::size_t, std::size_t>> observed, expected;
                sorting::for_each_bitonic_step(n, [&](std::size_t stage, std::size_t stride) {
                    expected.emplace_back(stage, stride);
                });
                auto step = [&](std::size_t stage, std::size_t stride, bool local) {
                    observed.emplace_back(stage, stride);
                    std::vector<unsigned> visits(n, 0);
                    for (std::size_t i = 0; i < n; ++i) {
                        const auto partner = i ^ stride;
                        require(partner < n, "Partner out of bounds");
                        require((i / tile == partner / tile) == local, "Wrong local/cross-tile routing");
                        if (partner > i) {
                            ++visits[i]; ++visits[partner];
                            if ((i & stage) == 0 ? output[i] > output[partner]
                                                 : output[i] < output[partner])
                                std::swap(output[i], output[partner]);
                        }
                    }
                    for (auto count : visits) require(count == 1, "Pair ownership is not unique");
                };
                sorting::for_each_shared_bitonic_launch(n,
                    [&](std::size_t actual_tile) {
                        require(actual_tile == tile, "Wrong tile selection");
                        for (std::size_t stage = 2; stage <= tile; stage *= 2)
                            for (std::size_t stride = stage / 2; stride; stride /= 2)
                                step(stage, stride, true);
                        for (std::size_t base = 0; base < n; base += tile) {
                            if ((base & tile) == 0)
                                require(std::is_sorted(output.begin() + base, output.begin() + base + tile), "Initial ascending tile");
                            else
                                require(std::is_sorted(output.begin() + base, output.begin() + base + tile,
                                                       std::greater<sorting::Key>()), "Initial descending tile");
                        }
                    },
                    [&](std::size_t stage, std::size_t stride) { step(stage, stride, false); },
                    [&](std::size_t stage, std::size_t actual_tile) {
                        require(actual_tile == tile, "Wrong finish tile");
                        for (std::size_t stride = tile / 2; stride; stride /= 2)
                            step(stage, stride, true);
                    });
                require(observed == expected, "Hybrid schedule differs from full Bitonic network");
                const auto result = sorting::check_sorted(input, output);
                if (!result.passed) throw std::runtime_error(result.message);
                ++cases;
            }
        }
        for (std::size_t n : {3u, 31u, 257u}) {
            bool invoked = false, rejected = false;
            try {
                sorting::for_each_shared_bitonic_launch(n,
                    [&](std::size_t) { invoked = true; },
                    [&](std::size_t, std::size_t) { invoked = true; },
                    [&](std::size_t, std::size_t) { invoked = true; });
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && !invoked, "Invalid size not rejected before launches");
        }
        std::cout << "PASS: CPU shared Bitonic logic: " << cases
                  << " cases; full network equivalence, alternating tiles, pair ownership, routing, results, and rejection verified. GPU NOT tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
