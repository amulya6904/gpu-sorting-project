#pragma once
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace sorting {
struct BenchmarkOptions;
struct TrialTimes { double h2d = 0, sort = 0, d2h = 0, wall = 0; };
inline double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("Median requires measurements");
    std::sort(values.begin(), values.end());
    const auto i = values.size() / 2;
    return values.size() % 2 ? values[i] : (values[i - 1] + values[i]) / 2;
}
inline TrialTimes medians(const std::vector<TrialTimes>& values) {
    std::vector<double> h, s, d, w;
    for (const auto& t : values) { h.push_back(t.h2d); s.push_back(t.sort); d.push_back(t.d2h); w.push_back(t.wall); }
    return {median(h), median(s), median(d), median(w)};
}
inline std::string csv_field(const std::string& value) {
    std::string result = "\"";
    for (char c : value) { result += c; if (c == '"') result += c; }
    return result + '"';
}
struct BenchmarkResult {
    std::string algorithm, status, reason;
    std::vector<TrialTimes> trials;
};
void write_benchmark_csv(const BenchmarkOptions& options, const std::string& gpu,
                         const std::string& build_configuration,
                         const std::vector<BenchmarkResult>& results);
} // namespace sorting
