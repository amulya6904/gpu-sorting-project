#include "benchmark_options.hpp"
#include "benchmark_results.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        auto require = [](bool ok) { if (!ok) throw std::runtime_error("CLI/support check failed"); };
        const auto defaults = sorting::parse_benchmark_options({});
        require(defaults.algorithms.size() == 7 && defaults.size == 1024 && defaults.trials == 10 && defaults.warmups == 3);
        const auto options = sorting::parse_benchmark_options({"--algorithm", "radix-1,radix-2,radix-4,radix-8,cub",
            "--size", "257", "--distribution", "duplicate-heavy", "--seed", "4294967295", "--warmups", "2", "--trials", "4"});
        require(options.algorithms.size() == 5 && options.size == 257 && options.seed == 4294967295u && options.trials == 4);
        require(options.distribution == sorting::Distribution::DuplicateHeavy && options.warmups == 2);
        for (const auto& algorithm : sorting::benchmark_algorithms()) {
            require(sorting::benchmark_supported(algorithm, 0) && sorting::benchmark_supported(algorithm, 1));
            require(sorting::benchmark_supported(algorithm, 1024));
            require(sorting::benchmark_supported(algorithm, 257) == (algorithm.find("bitonic") != 0));
        }
        for (const auto& name : {"random", "sorted", "reverse-sorted", "all-equal", "duplicate-heavy", "boundary-value"})
            sorting::parse_benchmark_options({"--distribution", name});
        for (const std::vector<std::string>& invalid : std::vector<std::vector<std::string>>{
            {"--algorithm", "oops"}, {"--algorithm", "cub,cub"}, {"--algorithm", "cub,"},
            {"--size", "-1"}, {"--size", "2147483648"}, {"--size", "12x"},
            {"--seed", "4294967296"}, {"--trials", "0"}, {"--warmups", "0"},
            {"--distribution", "oops"}, {"--size"}, {"--oops", "1"}}) {
            bool rejected = false;
            try { sorting::parse_benchmark_options(invalid); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected);
        }
        require(sorting::parse_benchmark_options({"--help"}).help);
        require(sorting::median({5, 1, 3}) == 3 && sorting::median({8, 2, 4, 6}) == 5);
        const auto metrics = sorting::medians({{1, 8, 2, 20}, {3, 2, 4, 10}});
        require(metrics.h2d == 2 && metrics.sort == 5 && metrics.d2h == 3 && metrics.wall == 15);
        require(sorting::csv_field("a,\"b") == "\"a,\"\"b\"");
        require(sorting::parse_benchmark_options({"--output-prefix", "case"}).output_prefix == "case");
        if (argc == 2) {
            auto fixture = defaults;
            fixture.output_prefix = argv[1]; fixture.trials = 2;
            sorting::write_benchmark_csv(fixture, "SYNTHETIC FIXTURE GPU", "Release", {
                {"radix-1", "PASS", "SYNTHETIC FIXTURE", {{1,4,2,10},{3,8,4,20}}},
                {"cub", "PASS", "SYNTHETIC FIXTURE", {{1,2,2,5},{3,4,4,10}}},
                {"bitonic-global", "SKIPPED", "SYNTHETIC skipped, with quote \"", {}},
                {"radix-2", "FAIL", "SYNTHETIC failure", {}}});
        }
        std::cout << "PASS: benchmark option parsing, selections, size contracts, and invalid arguments verified on CPU. GPU NOT tested.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
