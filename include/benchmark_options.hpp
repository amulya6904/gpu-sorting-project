#pragma once
#include "sorting_support.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace sorting {
inline std::vector<std::string> benchmark_algorithms() {
    return {"bitonic-global", "bitonic-shared", "radix-1", "radix-2", "radix-4", "radix-8", "cub"};
}
struct BenchmarkOptions {
    std::vector<std::string> algorithms = benchmark_algorithms();
    std::size_t size = 1024;
    Distribution distribution = Distribution::Random;
    std::string distribution_name = "random";
    Key seed = 12345;
    unsigned warmups = 3, trials = 10;
    std::string output_prefix;
    bool help = false;
};
inline std::uint64_t benchmark_number(const std::string& value, std::uint64_t maximum) {
    if (value.empty()) throw std::invalid_argument("Empty numeric argument");
    std::uint64_t result = 0;
    for (char c : value) {
        if (c < '0' || c > '9' || static_cast<unsigned>(c - '0') > maximum ||
            result > (maximum - static_cast<unsigned>(c - '0')) / 10)
            throw std::invalid_argument("Invalid or out-of-range number: " + value);
        result = result * 10 + static_cast<unsigned>(c - '0');
    }
    return result;
}
inline BenchmarkOptions parse_benchmark_options(const std::vector<std::string>& args) {
    BenchmarkOptions options;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& name = args[i];
        if (name == "--help") { options.help = true; continue; }
        if (i + 1 == args.size()) throw std::invalid_argument("Missing value for " + name);
        const auto& value = args[++i];
        if (name == "--output-prefix") {
            options.output_prefix = value;
        } else if (name == "--algorithm") {
            options.algorithms.clear();
            const auto available = benchmark_algorithms();
            if (value == "all") options.algorithms = available;
            else {
                std::size_t begin = 0;
                do {
                    const auto end = value.find(',', begin);
                    const auto algorithm = value.substr(begin, end - begin);
                    if (std::find(available.begin(), available.end(), algorithm) == available.end())
                        throw std::invalid_argument("Unknown algorithm: " + algorithm);
                    if (std::find(options.algorithms.begin(), options.algorithms.end(), algorithm) != options.algorithms.end())
                        throw std::invalid_argument("Duplicate algorithm: " + algorithm);
                    options.algorithms.push_back(algorithm);
                    if (end == std::string::npos) break;
                    begin = end + 1;
                } while (true);
            }
        } else if (name == "--size") {
            // Use the portable int-count CUB API, including older CUDA toolkits.
            options.size = static_cast<std::size_t>(benchmark_number(value, std::numeric_limits<int>::max()));
        } else if (name == "--seed") {
            options.seed = static_cast<Key>(benchmark_number(value, std::numeric_limits<Key>::max()));
        } else if (name == "--warmups" || name == "--trials") {
            const auto count = static_cast<unsigned>(benchmark_number(value, std::numeric_limits<unsigned>::max()));
            if (!count) throw std::invalid_argument("Warmups and trials must be positive");
            if (name == "--warmups") options.warmups = count;
            else options.trials = count;
        } else if (name == "--distribution") {
            options.distribution_name = value;
            if (value == "random") options.distribution = Distribution::Random;
            else if (value == "sorted") options.distribution = Distribution::Sorted;
            else if (value == "reverse-sorted") options.distribution = Distribution::ReverseSorted;
            else if (value == "all-equal") options.distribution = Distribution::AllEqual;
            else if (value == "duplicate-heavy") options.distribution = Distribution::DuplicateHeavy;
            else if (value == "boundary-value") options.distribution = Distribution::BoundaryValues;
            else throw std::invalid_argument("Unknown distribution: " + value);
        } else throw std::invalid_argument("Unknown option: " + name);
    }
    return options;
}
inline bool benchmark_supported(const std::string& algorithm, std::size_t n) {
    return algorithm.compare(0, 8, "bitonic-") != 0 || n < 2 || (n & (n - 1)) == 0;
}
} // namespace sorting
