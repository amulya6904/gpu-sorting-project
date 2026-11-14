#include "benchmark_options.hpp"
#include "benchmark_results.hpp"
namespace sorting {
void write_benchmark_csv(const sorting::BenchmarkOptions& options, const std::string& gpu, const std::string& build_configuration,
                    const std::vector<sorting::BenchmarkResult>& results) {
    if (options.output_prefix.empty()) return;
    std::ofstream raw(options.output_prefix + "_raw.csv"), summary(options.output_prefix + "_summary.csv");
    if (!raw || !summary) throw std::runtime_error("Cannot open CSV output prefix: " + options.output_prefix);
    const std::string common = "algorithm,key_count,distribution,seed,warmups,trials,gpu,host_memory,build_configuration,";
    raw << common << "trial,h2d_ms,sort_ms,d2h_ms,end_to_end_ms,status,reason\n";
    summary << common << "completed_trials,median_h2d_ms,median_sort_ms,median_d2h_ms,median_end_to_end_ms,keys_per_second,percentage_of_cub,status,reason\n";
    raw << std::setprecision(12); summary << std::setprecision(12);
    double cub_ms = 0;
    for (const auto& r : results)
        if (r.algorithm == "cub" && r.status == "PASS") cub_ms = sorting::medians(r.trials).sort;
    for (const auto& r : results) {
        const auto prefix = sorting::csv_field(r.algorithm) + ',' + std::to_string(options.size) + ','
            + sorting::csv_field(options.distribution_name) + ',' + std::to_string(options.seed) + ','
            + std::to_string(options.warmups) + ',' + std::to_string(options.trials) + ','
            + sorting::csv_field(gpu) + ",pinned," + build_configuration + ',';
        for (std::size_t i = 0; i < r.trials.size(); ++i) {
            const auto& t = r.trials[i];
            raw << prefix << i << ',' << t.h2d << ',' << t.sort << ',' << t.d2h << ',' << t.wall
                << ',' << r.status << ',' << sorting::csv_field(r.reason) << '\n';
        }
        if (r.status != "PASS") raw << prefix << ",,,,," << r.status << ',' << sorting::csv_field(r.reason) << '\n';
        summary << prefix << r.trials.size() << ',';
        if (r.status == "PASS") {
            const auto m = sorting::medians(r.trials);
            summary << m.h2d << ',' << m.sort << ',' << m.d2h << ',' << m.wall << ',';
            if (m.sort > 0) summary << options.size * 1000.0 / m.sort;
            summary << ',';
            if (m.sort > 0 && cub_ms > 0 && options.size > 0) summary << 100.0 * cub_ms / m.sort;
            summary << ',';
        } else summary << ",,,,,,";
        summary << r.status << ',' << sorting::csv_field(r.reason) << '\n';
    }
    raw.flush(); summary.flush();
    if (!raw || !summary) throw std::runtime_error("Failed writing benchmark CSVs");
}
}
