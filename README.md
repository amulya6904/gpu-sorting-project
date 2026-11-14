# CUDA GPU Sorting: benchmarking and kernel profiling

Phases 1–2 provide the CUDA vector-add smoke test; Phase 3 adds CPU-only dataset
generation and sorting correctness checks. Phase 4 adds global-memory CUDA
Bitonic Sort and its tests. Phase 5 adds the shared-memory version and a small
sort-only comparison; GPU verification and timings are pending on this laptop.
See [SETUP_STATUS.md](SETUP_STATUS.md) for Phase 2 diagnostics and
[PHASE4_RESULTS.md](PHASE4_RESULTS.md) for Phase 4 verification status.

## Phase 9: discovery-first kernel profiling

Actual status: [PHASE9_RESULTS.md](PHASE9_RESULTS.md). This laptop has neither
CUDA nor Nsight Compute, and Phase 8 contains no successful GPU measurements.
Hardware bottlenecks and optimization benefits remain pending. The prepared
workflow uses Radix-1/Radix-8 as the fallback extremes, not as measured winners.

`CUDA_LINEINFO=ON` adds `-lineinfo` to CUDA compilation without `-G` or changing
Release optimization. No sort implementation, default tile, or benchmark timing
behavior was changed for profiling. Normal runner invocations retain complete
verification. Capture runs intentionally profile selected kernels in the first
warmup, then allow execution to finish so correctness is still checked. Times
printed under ncu are **not Phase 8 benchmark times**.

### Local CPU checks: exact PowerShell commands

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G "Visual Studio 16 2019" -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
python tests\test_profile_workflow.py
python -W ignore::DeprecationWarning tests\test_benchmark_pipeline.py
```

The new Python tests independently enumerate the Bitonic schedules to validate
the selected launch indices, parse a labelled CSV test row, distinguish error
classes, and verify that a missing profiler produces no measured values.

### Windows NVIDIA lab machine: Release build and profiler discovery

Use an x64 Developer PowerShell with compatible CUDA/MSVC and CUDA integration:

```powershell
cmake -S . -B build-phase9 -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DCUDA_LINEINFO=ON -DBUILD_TESTING=ON
cmake --build build-phase9 --config Release --parallel
ctest --test-dir build-phase9 -C Release --output-on-failure
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
nvcc --version
ncu --version
ncu --list-sections
ncu --query-metrics --query-metrics-mode all --devices 0
python scripts\profile_kernels.py --runner build-phase9\Release\sort_benchmark.exe --output profiles\phase9_lab --size 65536 --seed 12345 --radix-widths 1 8
```

Stop if configure/build/correctness fails. For a supported VS 2019 installation,
substitute `Visual Studio 16 2019` in a new build directory. Use full paths when
CMake or ncu is outside PATH. To locate an installed Windows profiler:

```powershell
Get-ChildItem 'C:\Program Files\NVIDIA Corporation' -Filter ncu.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
```

Pass the selected executable as `--ncu 'C:\actual\installed\path\ncu.exe'` to
the script. If absent, the lab administrator must provide a compatible Nsight
Compute installation. Do not infer counter denial merely because ncu is missing.

### Linux NVIDIA lab machine

Obtain a GPU session and load the lab's CUDA/compiler/profiler modules:

```bash
cmake -S . -B build-phase9 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DCUDA_LINEINFO=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phase9 --parallel
ctest --test-dir build-phase9 --output-on-failure
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
nvcc --version
ncu --version
ncu --list-sections
ncu --query-metrics --query-metrics-mode all --devices 0
python3 scripts/profile_kernels.py --runner build-phase9/sort_benchmark --output profiles/phase9_lab --size 65536 --seed 12345 --radix-widths 1 8
```

Use a fresh output directory each time. `--size 262144` selects the other
representative workload; `--duplicate-heavy` adds six histogram/scan/scatter
captures using duplicate-heavy input. Use the same size, seed, distribution,
and device for direct comparisons. If Phase 8 measured results exist, explicitly
select two radix widths based on the matching workload and record that decision;
otherwise the documented default extremes remain appropriate.

### What is captured

The script saves profiler version/help, section listing, and full supported
metric names **before** selecting metrics. If supported by that version it also
queries source metrics separately. It selects only exact names reported by the
installed profiler for device 0; missing metrics remain explicitly UNAVAILABLE.
Section availability alone does not prove a GPU supports every counter.

After discovery it runs the unprofiled all-seven smoke. Each capture uses
`--kernel-name-base demangled`, a kernel-name regex, `--launch-skip N`, and
`--launch-count 1`. Nsight's skip count applies only to matching kernel names,
not all launches. No kill option is used, so the application finishes and checks
its output. The selection is deterministic in the unchanged sort schedules:

| Algorithm / kernel | At n=65536, matching launches skipped | Captured work, first warmup |
| --- | --- | --- |
| Global / bitonic_step | 120 | stage 65536, stride 32768; cross-tile |
| Global / bitonic_step | 28 | stage 256, stride 128; within-tile, still global memory |
| Shared / bitonic_step | 28 | stage 65536, stride 32768; cross-tile |
| Shared / tile_steps | 0 | initial stages 2 through 256, alternating tile runs |
| Shared / tile_steps | 8 | final stage 65536, remaining strides 128 through 1 |
| Radix-1 / histogram_step, prefix_step, scatter_step | 0 each | pass 0, shift 0, bits 1 |
| Radix-8 / histogram_step, prefix_step, scatter_step | 0 each | pass 0, shift 0, bits 8 |

At n=262144 the global final-stage skip is 153, shared cross-tile skip is 45,
and shared final tile skip is 10. The script records the computed selection.
Grouped tile kernels perform more than one network step, so their duration
must not be compared as if it were one global compare kernel.

One explicit Windows example after discovering supported metrics (replace the
metric list if these names are unavailable on the lab GPU):

```powershell
ncu --kernel-name-base demangled --kernel-name 'regex:.*bitonic_step.*' --launch-skip 120 --launch-count 1 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum --cache-control all --clock-control none --export profiles\manual_global_final.ncu-rep .\build-phase9\Release\sort_benchmark.exe --algorithm bitonic-global --size 65536 --distribution random --seed 12345 --warmups 1 --trials 1
ncu --import profiles\manual_global_final.ncu-rep --page raw --csv
```

The workflow uses kernel replay, cache-control all (consistent replay cache
policy), and clock-control none (no requested clock locking). Record these
conditions; profiling/replay overhead and cold-cache effects differ from normal
benchmark runs. See the [official CLI launch filters and metric discovery](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

### Counter access, reports, and evidence limits

The first actual capture tests counter access. `ERR_NVGPUCTRPERM` is recorded as
COUNTER_PERMISSION_DENIED, distinct from MISSING_EXECUTABLE, NO_SUPPORTED_GPU,
or another tool/runtime error. Preserve the exact log. Ask the lab administrator
to use [NVIDIA's counter-access instructions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
when that error actually occurs; this workflow changes no permissions or system
settings. No such denial was observed on this laptop because ncu is absent.

`profiles/phase9_lab/` contains `.ncu-rep` files, raw imported CSV logs, discovery
logs, metadata with GPU/driver/toolkit/profiler query outputs and exact commands,
and `summary.csv` with algorithm/kernel/workload/capture/metric/value/unit/status/
interpretation/report. Failed discovery creates only logs and UNAVAILABLE rows,
never synthetic counter values. Each capture is bounded to one matching launch;
additional replay passes may be required to collect selected counters.

| Evidence | What it can support / limitation |
| --- | --- |
| DRAM throughput and read/write bytes | Direct device-memory traffic/throughput; low throughput alone does not establish a bandwidth bottleneck |
| Global load sectors and requests | Measured transaction traffic; sectors/request is a transaction-footprint proxy, not universal access efficiency |
| Shared load/store bank-conflict counters | Direct reported conflicts; interpret alongside access volume and distinguish shared atomic serialization |
| Thread instructions per warp instruction | Active-thread proxy; low values may come from bounds, pair ownership, predication, loop tails, or divergence |
| Uniform branch-target percentage | Branch-behavior proxy; does not capture all predication or explain why lanes are inactive |

Source-level hypotheses to test, **not measured conclusions**: Bitonic pair
ownership leaves half the logical indices without compare work; shared kernels
trade repeated global accesses for barriers/shared accesses; the radix scan has
one block and only one active digit thread per bucket; scatter computes ranks by
looping over earlier tile indices; duplicate-heavy histograms may serialize
shared atomic updates. Use measured counters and unprofiled times to decide
which, if any, is the limiting cost. Missing counters do not imply zero traffic,
zero conflicts, or no divergence.

### Optimization decision and controlled verification

No optimization was applied: there is no measured bottleneck on this machine.
Retain existing defaults. On the lab GPU, first save a before snapshot of the
source, build flags, reports, and unprofiled CSVs. Pick at most one small change
supported by those reports, state the expected affected metric, then rebuild
Release with the same lineinfo/toolchain/architecture settings. Rerun CTest,
including radix stable-scatter tests if affected. Example regular-run timing,
outside ncu, before and after using new prefixes and separate build directories:

```powershell
.\build-phase9\Release\sort_benchmark.exe --algorithm bitonic-global,bitonic-shared,radix-1,radix-8,cub --size 65536 --distribution random --seed 12345 --warmups 3 --trials 10 --output-prefix profiles\before_unprofiled
```

Use the corresponding after executable and `after_unprofiled` prefix, rerun the
affected one-launch capture with the same filters/cache policy, and repeat paired
unprofiled runs to assess repeatability. Ensure any code change that alters
launch counts also updates and validates capture mapping. Report a flat or worse
outcome honestly; keep the existing default unless correctness passes and a
repeatable unprofiled benefit is demonstrated. A source hypothesis or isolated
profiler duration is not evidence of a speedup.

Phase 9 completes only when real profiling explains representative kernels'
memory/thread behavior. A speedup is optional. Phase 10 packaging is not started.

## Phase 8: reproducible matrix, CSVs, and charts

The runner now measures separate H2D, full-sort, and D2H CUDA-event intervals,
plus end-to-end steady-clock wall time. `scripts/run_benchmarks.py` orchestrates
the existing runner; it first requires all seven algorithms to pass the same
1,024-key random smoke case, then launches the configured matrix. It never
substitutes CPU timings for GPU measurements. See [PHASE8_RESULTS.md](PHASE8_RESULTS.md)
for actual local checks and the current hardware blocker.

Defaults: seven algorithms; 1,024, 65,536, 262,144, and 1,048,576 keys;
random, sorted, reverse-sorted, and duplicate-heavy inputs; seed 12345;
3 warmups and 10 measured trials. That is 112 algorithm cases and 1,120 measured
trials, plus warmups and the preliminary smoke. The matrix script supports
`--sizes`, `--distributions`, `--seeds`, `--algorithms`, `--warmups`, and `--trials`.
Lists use space-separated values. The underlying runner retains its Phase 7
single-case CLI, now with defaults 3/10 and `--output-prefix PATH` for CSV export.

### Measurement policy

- Use a **Release** executable; the runner rejects other build configurations.
- One seeded original is generated per size/distribution/seed. All algorithms
  and all trials restore that same input order. Algorithms execute in the
  configured order on device 0; no claim of clock/frequency control is made.
- All algorithms use reusable **pinned host memory** allocated with
  `cudaMallocHost`. Original keys are staged there once before warmup. Each
  trial performs its measured H2D restore and D2H result copy asynchronously on
  the same stream as the sort, using four boundary events.
- The sort interval includes every required GPU kernel, all radix passes,
  histogram resets inside the kernels, and the complete CUB operation. One-time
  device/host allocations, workspace queries/setup, generation, and verification
  are outside. Launch submission gaps can appear in CUDA-event intervals.
- Wall time starts immediately before the H2D copy API call and ends after the
  output-copy stream wait. It includes host launch/scheduling and wait overhead,
  but excludes generation, setup, and CPU verification. It is a separate clock
  measurement, not the sum of the event medians. The CPU copy from pinned output
  to the checker vector and std::sort checking happen after wall timing stops.
- Every warmup and trial is fully verified. Failed/skipped cases retain reasons
  and are excluded from comparisons. After a configuration failure the runner
  skips the remaining configurations in that process rather than reusing a
  potentially broken CUDA context; later workload cases use fresh processes.
- `keys_per_second = n * 1000 / median_sort_ms`.
  `percentage_of_cub = 100 * custom_keys_per_second / matching_cub_keys_per_second`.
  Matching means the same size, distribution, seed, and GPU. Missing/failed CUB
  or zero-resolution sort measurements leave percentages blank.

### CPU build and local validation: PowerShell

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G "Visual Studio 16 2019" -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
python -m pip install -r scripts\requirements.txt
python -W ignore::DeprecationWarning tests\test_benchmark_pipeline.py
```

The existing laptop already had Python 3.12.10 and Matplotlib 3.8.1; no package
installation was needed for the recorded checks. Python tests validate plot
generation, median/throughput/CUB calculations against raw records, failure
filtering, corruption/missing-trial rejection, and the smoke gate. C++ tests
verify medians, CSV quoting, and CLI options. CPU-buildable CSV export is reused
unchanged by the CUDA runner.

### Release build and real matrix: Windows NVIDIA lab machine

From an x64 Developer PowerShell with compatible CUDA/MSVC and CUDA integration
(VS 2022 example; use VS 2019's generator if appropriate):

```powershell
cmake -S . -B build-phase8 -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DBUILD_TESTING=ON
cmake --build build-phase8 --config Release --parallel
ctest --test-dir build-phase8 -C Release --output-on-failure
python -m pip install -r scripts\requirements.txt
python scripts\run_benchmarks.py --runner build-phase8\Release\sort_benchmark.exe --output results\phase8_lab --warmups 3 --trials 10
python scripts\plot_benchmarks.py --input results\phase8_lab --output results\phase8_lab\charts
```

Stop on any failure before advancing to plotting. The matrix driver repeats the
all-seven smoke internally and refuses to start the matrix unless it passes.
Use a new output directory for each run; existing directories are never
overwritten. If CMake is outside PATH, invoke it using the full-path pattern
above. Neither CUB-disabled builds nor failed smoke tests count as completion.

### Release build and real matrix: Linux NVIDIA lab machine

Obtain a GPU session and load the lab's compatible CUDA/compiler modules:

```bash
cmake -S . -B build-phase8 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phase8 --parallel
ctest --test-dir build-phase8 --output-on-failure
python3 -m pip install -r scripts/requirements.txt
python3 scripts/run_benchmarks.py --runner build-phase8/sort_benchmark --output results/phase8_lab --warmups 3 --trials 10
python3 scripts/plot_benchmarks.py --input results/phase8_lab --output results/phase8_lab/charts
```

For example, append `--sizes 1024 65536 --distributions random duplicate-heavy
--seeds 12345 67890` to the matrix command for a smaller reproducible matrix.
The defaults include all required core sizes and distributions; a smaller matrix
does not fulfill the full Phase 8 measurement requirement.

### Output files and interpretation

- `raw.csv`: measured per-trial H2D/sort/D2H/wall milliseconds, configuration,
  seed, requested warmups/trials, GPU, host-memory/build policy, status/reason.
  Warmups are verified but excluded from raw measured-trial records.
- `summary.csv`: completed trial count, medians of each interval, keys/s,
  percentage of matching CUB, and status/reason. Failed summaries have blank
  performance metrics even if some trials completed before failure.
- `metadata.json`: GPU/UUID/driver query via nvidia-smi, nvcc version output,
  runtime/header and driver API versions in runner logs, OS, required Release
  build, host-memory policy, matrix, UTC timestamp, and exact command argument
  arrays with stdout/stderr. These arrays can be re-executed without ambiguity.
- `cases/`: each workload's exact log and per-case CSVs. `smoke.log` records the
  prerequisite all-seven test. A blocked smoke exports skipped-case summary rows
  and a header-only raw CSV; these are a status record, not measurements.
- `charts/throughput_*.png`: all seven throughputs versus size, panels per
  distribution and separate figures per GPU/seed.
- `charts/radix_widths_*.png`: radix digit-width throughput comparison.
- `charts/event_breakdown_*.png`: stacked median H2D/sort/D2H at the smallest and
  largest available sizes, explicitly labelled **CUDA event breakdown**.
- `charts/analysis.txt`: fastest custom configuration per workload, percentage of
  matching CUB, event transfer fraction, and separately labelled wall medians.

Plotting checks successful raw trial counts, medians, throughput, and CUB
percentages before drawing. It excludes failed/skipped cases and sort times
below event resolution. Stacked component medians are not median wall time.

### Synthetic plotting fixtures (not measurements)

Fixtures live only under `tests/fixtures/`, separate from `results/`. The
metadata and every fixture chart explicitly say **SYNTHETIC FIXTURE - NOT GPU
MEASUREMENTS**. They must not be used to infer a winner or speedup.

```powershell
python tests\test_benchmark_pipeline.py --write-fixture tests\fixtures\phase8_synthetic
python scripts\plot_benchmarks.py --input tests\fixtures\phase8_synthetic --output tests\fixtures\phase8_synthetic\charts --fixture
```

Phase 8 remains incomplete until the required matrix runs on real CUDA hardware
and these charts are produced from the measured CSVs. Phase 9 profiling and
kernel optimization are outside this phase.

## Phase 7: unified runner and CUB baseline

`sort_benchmark` selects `bitonic-global`, `bitonic-shared`, `radix-1`, `radix-2`,
`radix-4`, `radix-8`, and `cub`. Use `--algorithm all`, one name, or a comma-separated
subset. Options are `--size`, `--distribution`, `--seed`, `--warmups`, and `--trials`.
Defaults are all, 1024, random, 12345, 1 warmup, and 3 trials. Distributions are
`random`, `sorted`, `reverse-sorted`, `all-equal`, `duplicate-heavy`, and
`boundary-value`. Sizes 0 through INT_MAX are accepted; the limit preserves
compatibility with the int-count CUB API in older CUDA toolkits. Warmups/trials
must be positive. `--help` prints usage without initializing CUDA.

The runner generates exactly one original input for a case. Every algorithm,
warmup, and measured trial restores those exact keys and waits for restoration
before the start event. Events on the single sorting stream bracket the complete
sort. All allocations, CUB storage queries, input restoration, copies, and full
std::sort-based verification are outside the interval. Every warmup and trial
is verified; a mismatch or CUDA error makes the run fail. Each configuration
reports median sort-only milliseconds (averaging middle values for even trial
counts). Event timing may include device idle gaps during host submission.
This runner currently executes one small case, not a full benchmark matrix.

CUB is included only inside the CUDA-enabled runner, using
`<cub/device/device_radix_sort.cuh>` from the selected toolkit, without downloads
or vendored headers. The runner requests ascending SortKeys over bits `[0,32)`
on its stream. It uses the pointer overload with separate, non-overlapping input
and output allocations, queries temporary bytes before warmup, and reuses the
allocated storage for all trials of that case. It copies back CUB's explicit
output buffer, while custom full sorts return their original input buffer.
Empty inputs are handled as a no-op. See the
[NVIDIA CUB API source and documentation](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh)
for the query and pointer-overload requirements. The runner requests CUDA C++17
to accommodate modern toolkit headers; custom libraries retain their C++14 setup.

### CPU regression tests on this Windows laptop

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G "Visual Studio 16 2019" -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\benchmark_options_tests.exe
```

No CUDA or CUB is required. This retains all prior CPU tests and adds option
validation and Bitonic size-contract checks. It does not verify GPU dispatch.

### Windows NVIDIA lab machine: configure, build, and smoke tests

From an x64 Developer PowerShell with a compatible toolkit/MSVC, CUDA Visual
Studio integration, and CMake in PATH (VS 2022 example):

```powershell
cmake -S . -B build-phase7 -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DBUILD_TESTING=ON
cmake --build build-phase7 --config Release --parallel
ctest --test-dir build-phase7 -C Release --output-on-failure
.\build-phase7\Release\sort_benchmark.exe --algorithm all --size 1024 --distribution random --seed 12345 --warmups 1 --trials 3
.\build-phase7\Release\sort_benchmark.exe --algorithm radix-1,radix-2,radix-4,radix-8,cub --size 257 --distribution duplicate-heavy --seed 12345 --warmups 1 --trials 3
```

For VS 2019 use `Visual Studio 16 2019` with a compatible toolkit and new build
directory; invoke CMake by full path if absent from PATH. Stop after any failing
command. `BUILD_TESTING=ON` registers both runner smoke cases in CTest alongside
the prior tests. Direct runs display successful configuration names and measured
times; CTest normally hides successful output. The runner also builds with
`BUILD_TESTING=OFF`.

To demonstrate unsupported sizes without preventing Radix/CUB from running:

```powershell
.\build-phase7\Release\sort_benchmark.exe --algorithm all --size 257 --distribution duplicate-heavy --seed 12345 --warmups 1 --trials 3
```

This reports both Bitonic entries as UNSUPPORTED, then verifies the other five.
Unsupported sizes are not silently padded. If all selected entries are
unsupported the runner returns 2; otherwise verified supported entries can
succeed with unsupported entries reported. CUDA errors, mismatches, invalid
arguments, or unavailable requested CUB return 1.

### Linux NVIDIA lab machine

Obtain a GPU session and load compatible CUDA/compiler modules first:

```bash
cmake -S . -B build-phase7 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phase7 --parallel
ctest --test-dir build-phase7 --output-on-failure
./build-phase7/sort_benchmark --algorithm all --size 1024 --distribution random --seed 12345 --warmups 1 --trials 3
./build-phase7/sort_benchmark --algorithm radix-1,radix-2,radix-4,radix-8,cub --size 257 --distribution duplicate-heavy --seed 12345 --warmups 1 --trials 3
```

### If CUB headers are missing or incompatible

Preserve the exact compiler error and verify the CUDA toolkit installation and
its supported host compiler. No separate CUB installation is attempted here.
To continue with the six custom configurations, configure a separate build with
`-DENABLE_CUB_BASELINE=OFF`, build, and select
`--algorithm bitonic-global,bitonic-shared,radix-1,radix-2,radix-4,radix-8`.
This excludes all CUB includes and API calls from the runner. Selecting cub/all
in that build reports CUB as UNAVAILABLE and returns failure after running other
supported selections; it cannot count as all-seven verification. CUB-dependent
CTest smoke cases are only registered when CUB is enabled.

The current laptop fails at CUDA toolchain discovery before any CUB header can
be compiled. See [PHASE7_RESULTS.md](PHASE7_RESULTS.md) for actual results.
Phase 7 is GPU-verified only when all seven configurations pass the same-input
1,024-key smoke case on an actual CUDA-capable GPU. GPU results and timings
remain pending here. Phase 8's full matrix, transfer breakdowns, throughput
comparisons, and charts are not included.

## Phase 6: stable LSD Radix Sort

The parameterized `sorting::radix_sort` in `radix_sort.cuh` supports 1, 2, 4,
and 8 bits per pass. It handles arbitrary sizes, including a partial 256-key
tile. Sizes 0 and 1 are no-ops. Unsupported widths are rejected before any
kernel launch, including for empty inputs. Both Bitonic implementations remain
available unchanged through their existing entry points.

| Bits per pass | Buckets | Passes for uint32 keys | Kernel launches |
| --- | --- | --- | --- |
| 1 | 2 | 32 | 96 |
| 2 | 4 | 16 | 48 |
| 4 | 16 | 8 | 24 |
| 8 | 256 | 4 | 12 |

These launch counts apply to n>1. Wider digits reduce pass count but increase
histogram/offset storage and bucket work. Performance conclusions are pending
measured benchmarks; this phase prioritizes correctness.

Each pass extracts `(key >> shift) & ((1u << bits) - 1)` and queues three kernels
on the caller's stream:

1. **Histogram:** each input tile counts keys per digit. Shared atomic increments
   are used only for counts, where arrival order does not matter. The result is
   stored digit-major as `histogram[digit * tiles + tile]`.
2. **Exclusive prefix:** one GPU thread per digit scans tiles in input order,
   producing earlier-tile counts. A shared barrier publishes digit totals; each
   digit sums lower digit totals to get its global bucket start. The offset array
   stores bucket start plus earlier-tile count. This is a self-implemented,
   correctness-first GPU scan, with a serial tile loop per digit; no scan primitive
   or library sort is used. This scan is not yet optimized for large workloads.
3. **Stable scatter:** each valid thread counts earlier local indices with the
   same digit and writes to `offsets[digit * tiles + tile] + local_rank`.
   No atomic scatter positions are used. Earlier tiles precede later tiles, and
   within a tile original input order is preserved. Inactive threads in the last
   tile still participate in the shared-memory barriers.

Stability is required at every LSD pass. In decimal as a small analogy, the
ones-digit pass turns `[21, 12, 11, 22]` into `[21, 11, 12, 22]`. A stable tens
pass produces `[11, 12, 21, 22]`, preserving the ones order within each tens
group. An unstable tens pass could produce `[12, 11, 22, 21]`, undoing the earlier
work. The CUDA implementation applies the same rule to binary digit groups.

### Allocation-free sorting interface

Link the `radix_sort` CMake target and include `radix_sort.cuh`. The caller
allocates all device memory before calling the sort:

- `keys`: n uint32 keys, containing the original input.
- `workspace.alternate`: another n-key buffer, with `alternate_capacity >= n`.
- `workspace.histogram` and `workspace.offsets`: separate size_t arrays, each
  with at least `radix_workspace_entries(n,bits)` entries; set `workspace.entries`
  to the capacity of each array. Query workspace with the intended width, or use
  width 8 to reserve capacity for all four configurations at that n.

All allocations must be disjoint, remain alive until the stream completes, and
match the declared capacities. Check allocation and copy errors in the caller.
Then call:

```cpp
auto* result = sorting::radix_sort(keys, n, bits, workspace, stream);
// Check cudaStreamSynchronize(stream) to surface asynchronous execution errors.
// Copy n keys from result back to the CPU and use sorting::check_sorted.
```

The key buffers swap roles after each pass. All supported pass counts are even,
so the returned final pointer equals `keys`; the alternate holds the penultimate
pass. No allocation, host/device copy, or wait occurs inside the sort. Reuse the
workspace on the same ordered stream; concurrent sorts require separate storage.
`radix_pass` exposes a single stable out-of-place digit pass for tests. Its
aligned shift must be in [0,32-bits]. For n<2 it returns input without writing
output; otherwise it returns output. Launch errors throw; execution errors must
be checked by waiting on the stream. Empty/singleton full sorts need no workspace.

### CPU checks on this Windows laptop

From the project folder, exact PowerShell commands using the installed tools:

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G "Visual Studio 16 2019" -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\radix_logic_tests.exe
```

CTest retains all prior CPU suites and adds `radix_cpu_logic`. The CPU model
checks histogram/offset/rank equations, complete results, scatter destination
uniqueness, pass counts, workspace sizes, invalid widths, and single-pass
stability. It does not compile or execute the CUDA kernels.

### CUDA tests: Windows NVIDIA lab machine

Use an x64 Developer PowerShell with a compatible toolkit/MSVC and CUDA Visual
Studio integration. VS 2022 example (use VS 2019's generator instead if that is
the lab's supported toolchain):

```powershell
cmake -S . -B build-phase6 -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build-phase6 --config Release --parallel
ctest --test-dir build-phase6 -C Release --output-on-failure
.\build-phase6\Release\radix_gpu_tests.exe
```

Use the full-path CMake invocation above if CMake is not on PATH. Stop if any
step fails. CTest runs both Bitonic GPU suites and the new `radix_gpu_correctness`
test along with CPU suites. The direct radix command displays successful output.

### CUDA tests: Linux NVIDIA lab machine

Obtain a GPU session and load compatible CUDA/compiler modules first:

```bash
cmake -S . -B build-phase6 -DENABLE_CUDA=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phase6 --parallel
ctest --test-dir build-phase6 --output-on-failure
./build-phase6/radix_gpu_tests
```

The radix suites cover six distributions, four widths, and sizes 0, 1, 2, 31,
255, 256, 257, 511, 513, and 65,536 (240 full-sort cases). They reuse the Phase 3
generator and compare every result with std::sort. Sixteen targeted single-pass
cases use 769 distinct keys across multiple tiles, testing both mixed digits
and all-same digits at the low and high digit positions for each width. The
output is compared element by element with CPU `std::stable_sort` by that digit,
so changing the order of distinguishable equal-digit keys fails. Full key-only
sorting alone could not reveal reordered identical keys.

Actual results and blockers are recorded in [PHASE6_RESULTS.md](PHASE6_RESULTS.md).
Phase 6 is GPU-verified only after all four variants and stable-scatter cases
pass on an actual CUDA-capable GPU. No Phase 7 baseline integration or full
benchmarking is included.

## Phase 5: shared-memory Bitonic Sort

Call `sorting::bitonic_sort_shared(device_keys, n, stream)` from the same
`bitonic_sort.cuh` interface and link `bitonic_sort`. The original
`sorting::bitonic_sort` remains independently callable. Both sort unsigned
32-bit keys ascending in place, take the same caller-owned stream, and leave
allocation/transfers/waiting to the caller. Sizes 0 and 1 are no-ops; other sizes
must be powers of two. Invalid sizes are rejected before any kernel launch;
null pointers are rejected when n>1. No padding is used.

The default tile holds 256 keys: one key per thread, 256 threads and 1 KiB shared
memory per block. Inputs below 256 use a tile equal to n. These resource sizes
are within CUDA-capable device limits; the test driver also checks the device
properties before running. All tiles are full, so there are no inactive threads
skipping a required barrier. Each block loads a tile, synchronizes, runs the
local network with one thread per pair and a barrier after each step, then
writes it back. Blocks can process additional tiles with a uniform block-stride
loop; a final barrier protects shared-memory reuse.

Initial stages up to the tile size stay in shared memory. Direction uses the
global index `(i & stage) == 0`, so the final initial tiles alternate ascending
and descending. Adjacent opposing runs form the Bitonic sequences needed by the
next merge; sorting every initial tile ascending would break that network.
For later merge stages, strides >=256 cross tile boundaries and use the existing
global kernel. Once the stride falls below 256, the remaining comparisons fit
within each tile and run together in one shared-memory kernel. That kernel
preserves the current merge stage's direction rather than restarting a local
sort. Ordered launches on one stream separate dependent cross-block steps.
`__syncthreads()` separates dependent shared-memory steps within a block; every
thread reaches the barriers. See [NVIDIA shared-memory synchronization](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/).

### Exact CPU checks: Windows PowerShell

From the project folder using the laptop's existing tools (no CUDA required):

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G "Visual Studio 16 2019" -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\bitonic_shared_logic_tests.exe
```

CTest retains the Phase 3 and Phase 4 CPU tests and adds
`bitonic_shared_cpu_logic`. The new CPU simulation checks the hybrid schedule
against every step of the full network, local/cross-tile routing, alternating
initial tiles, unique pair ownership, invalid-size rejection, and entire sorted
results. CPU checks do not verify CUDA compilation, barriers, or GPU execution.

### GPU correctness and timing: Windows NVIDIA lab machine

Use an x64 Developer PowerShell with compatible MSVC, CUDA integration, and
CMake available. VS 2022 example:

```powershell
cmake -S . -B build-phase5 -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build-phase5 --config Release --parallel
ctest --test-dir build-phase5 -C Release --output-on-failure
.\build-phase5\Release\bitonic_gpu_tests.exe
.\build-phase5\Release\bitonic_shared_gpu_tests.exe
.\build-phase5\Release\bitonic_compare.exe
```

On a compatible VS 2019 machine substitute `Visual Studio 16 2019` and use a
new build directory. When CMake is outside PATH, use the full-path invocation
above. Stop after any failing configure/build/test command. Both GPU correctness
suites are registered in CTest; direct execution also displays successful output.
The comparison is built with BUILD_TESTING=ON but intentionally runs manually.

### GPU correctness and timing: Linux NVIDIA lab machine

Obtain a GPU session and load the lab's compatible CUDA/compiler modules first:

```bash
cmake -S . -B build-phase5 -DENABLE_CUDA=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phase5 --parallel
ctest --test-dir build-phase5 --output-on-failure
./build-phase5/bitonic_gpu_tests
./build-phase5/bitonic_shared_gpu_tests
./build-phase5/bitonic_compare
```

The shared GPU suite compares all outputs with the Phase 3 std::sort reference
for six distributions at 0, 1, 2, 4, 8, 32, 128, 256, 512, 1,024, 2,048, and
65,536 keys (72 cases), plus no-op, null-pointer, and unsupported-size checks.
This covers smaller tiles, one tile, two tiles, and multiple cross-tile merge
stages. GPU tests fail rather than silently skip when no GPU is available.

`bitonic_compare` uses identical seeded random inputs (seed 12345) for both
versions at n=256, 4096, and 65536. It restores input before every run, warms each
version twice, alternates execution order across seven trials per version,
and reports median global/shared milliseconds plus global/shared speedup.
Events on the sorting stream bracket the full kernel sequence only; allocation,
input restoration, copies, and correctness checks are outside the interval.
Every warmup and trial is checked against std::sort. A ratio below 1 means shared
memory was slower. This small comparison is not full benchmarking or profiling.
Event timing may include GPU idle gaps while the host submits the kernel sequence.

See [PHASE5_RESULTS.md](PHASE5_RESULTS.md) for actual checks and pending hardware
results. Phase 5 is GPU-verified only after shared correctness passes on an actual
CUDA-capable GPU. No timing or speedup is claimed before hardware measurement.

## Phase 4: global-memory Bitonic Sort

`sorting::bitonic_sort(device_keys, n, stream)` in `include/bitonic_sort.cuh`
sorts `std::uint32_t` keys in place, ascending. It enqueues kernels without
allocating memory, copying data, or waiting. The caller owns the allocation and
must wait on the same stream before consuming results or freeing memory.
Launch errors throw `std::runtime_error`; asynchronous execution errors are
reported by the caller's checked stream wait. Sizes 0/1 do nothing; larger sizes
must be powers of two. Invalid sizes throw `std::invalid_argument` before any
kernel launch. Null pointers are rejected for sizes greater than one. Padding
is not implemented; this Phase 4 entry point uses only global memory.

At each stage, `i ^ stride` flips one bit in the index to find its partner.
Applying XOR again returns `i`, so the pairing is symmetric. Only the thread
with `partner > i` reads and updates the pair, giving each key exactly one writer
per step. `(i & stage) == 0` selects ascending comparisons; the other groups use
descending comparisons to form the intermediate Bitonic sequences. At the final
stage every pair is ascending. Each step needs the preceding step's results
across all blocks, so it has its own kernel launch. All launches use the same
stream, whose normal execution ordering provides the dependency between steps;
no per-block barrier can synchronize the entire grid. See
[NVIDIA stream ordering](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/02-basics/asynchronous-execution.html).

### CPU logic verification on this laptop

Run the Phase 3 CPU-only configure/build/CTest commands below. CTest now runs
`cpu_correctness`, `bitonic_cpu_logic`, and (since Phase 5)
`bitonic_shared_cpu_logic`. For detailed Phase 4 logic-check output:

```powershell
.\build-cpu\Release\bitonic_logic_tests.exe
```

This simulates the host stage schedule and partner comparisons on the CPU,
checks unique ownership and the complete results against the Phase 3 reference,
and rejects 3, 31, and 257 before executing steps. It does not compile or execute
the CUDA kernel and cannot establish GPU correctness.

### CUDA-capable Windows lab machine

From a Developer PowerShell with CMake, a supported MSVC compiler, CUDA, and
Visual Studio CUDA integration installed (VS 2022 example):

```powershell
cmake -S . -B build-bitonic -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build-bitonic --config Release --parallel
ctest --test-dir build-bitonic -C Release --output-on-failure
.\build-bitonic\Release\bitonic_gpu_tests.exe
```

For VS 2019 use `-G "Visual Studio 16 2019"` with a compatible toolkit and new
build directory. On this laptop, the exact attempted configure command is in
PHASE4_RESULTS.md. The executable also prints GPU model and smoke timing; CTest
normally hides successful test output. Missing GPUs fail the test rather than
being silently skipped.

### CUDA-capable Linux lab machine

After loading the lab's CUDA/compiler modules and obtaining a GPU session:

```bash
cmake -S . -B build-bitonic -DENABLE_CUDA=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-bitonic --parallel
ctest --test-dir build-bitonic --output-on-failure
./build-bitonic/bitonic_gpu_tests
```

Use the machine-specific compiler/architecture settings documented below as
needed. The original `vector_add` target is still built when CUDA is enabled.

The GPU suite covers all six distributions at sizes 0, 1, 2, 4, 8, 16, 32, 256,
1,024, and 65,536, comparing every result with the existing std::sort reference.
It also tests the no-op API and rejection of 257. One random 65,536-key case
records CUDA events on the sorting stream around the full 136-kernel sequence;
allocation and host/device copies are outside that interval. This is a single
smoke measurement, not a benchmark. Expect a `PASS: GPU Bitonic correctness: 60
cases; ...` line and an actual measured time only after a successful GPU run.
Phase 4 is GPU-verified only after this suite passes on a CUDA-capable GPU.

## Phase 3: CPU-only tests on this Windows laptop

Only MSVC and CMake are required; no NVIDIA GPU, driver, toolkit, Python, or
Nsight is needed. From this project folder in ordinary PowerShell, use the
installed Visual Studio 2019 tools by full path (no PATH changes):

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G "Visual Studio 16 2019" -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\cpu_tests.exe
```

Stop if any command fails. The last command prints the detailed test summary;
CTest reports the registered suite. To use another arbitrary input size, add it
to the `sizes` array in `tests/cpu_tests.cpp` and rebuild. Sizes are not restricted
to powers of two: the suite includes 0, 1, 2, 4, 8, 16, 31, 257, and 65,536.
All six distributions are checked at every size, along with deliberate invalid
outputs to verify checker rejection. See [PHASE3_RESULTS.md](PHASE3_RESULTS.md)
for actual results.

On Linux with a C++ compiler and CMake:

```bash
cmake -S . -B build-cpu -DENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

`ENABLE_CUDA` defaults to `ON`, preserving the CUDA build commands below. Use
separate build directories for CPU-only and CUDA builds.

### Reusing the generator and checker

Link future C++ targets to `sorting_support` and include `sorting_support.hpp`.
Call `sorting::generate(size, distribution, seed)` for in-memory `std::uint32_t`
keys. The six distributions are `Random`, `Sorted`, `ReverseSorted`, `AllEqual`,
`DuplicateHeavy`, and `BoundaryValues`. Random, sorted, and reverse-sorted use
the same seeded mt19937 samples. All-equal uses 42; duplicate-heavy shuffles
repeated keys 0–7 with the seed; boundary values cycle through 0, UINT32_MAX,
1, UINT32_MAX−1, 2^31, and 2^31−1. Deterministic distributions ignore the seed.
The shuffle is deterministic and intended for duplicate stress cases, not
statistically uniform sampling. No files or CUDA calls are used by this library.

```cpp
auto original = sorting::generate(257, sorting::Distribution::Random, 12345);
auto candidate = original;
// Pass candidate to a sorting implementation; later copy GPU output back here.
auto result = sorting::check_sorted(original, candidate);
if (!result.passed) { /* report result.message */ }
```

The example's candidate is still unsorted until an algorithm fills it. The
checker sorts a separate copy of the original with `std::sort` and compares
every key, reporting a size mismatch or the first differing index and values.
It catches incorrect order, missing elements, wrong values, and duplicate-count
errors. Keep the original unchanged and give every future algorithm a copy of
the same generated input. Phase 4 can select power-of-two sizes without changing
the reusable generator/checker API.

## Prerequisites

- A CUDA-capable NVIDIA GPU for execution, with a compatible NVIDIA driver.
- NVIDIA CUDA Toolkit (`nvcc`) supporting the target GPU and host compiler.
- Windows: supported x64 MSVC C++ Build Tools and Windows SDK. Linux: a GCC/G++
  version supported by the selected toolkit and distribution.
- CMake 3.18 or newer (a recent version is recommended for recent toolkits).
- Python and Nsight Compute are optional for this smoke test.

An Intel Iris Xe GPU cannot execute this CUDA program. Installing the toolkit
alone will not enable GPU execution on the current laptop. Use a college lab or
teammate's NVIDIA machine for the remaining verification.

## Preparing a Windows NVIDIA machine

1. Run `nvidia-smi` and record the GPU and driver version. Check that the GPU is
   supported by the chosen [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
2. In Visual Studio Installer, install/modify Build Tools: select **Desktop
   development with C++**, the toolkit-supported x64 MSVC toolset, a Windows SDK,
   and **C++ CMake tools for Windows**. Install Visual Studio before CUDA.
3. Follow the selected toolkit's [official Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
   Install the CUDA compiler/runtime and Visual Studio integration. Install or
   update the NVIDIA driver as required by that release and the GPU, then reboot
   if requested. Nsight Compute can be installed for later profiling.
4. Open a fresh x64 Developer PowerShell. Confirm `cl`, `cmake --version`,
   `nvcc --version`, and `nvidia-smi` work. Add CMake to PATH via its installer
   if needed, or invoke its full path. Check the toolkit's compiler support table;
   do not assume every MSVC update is supported.

The existing laptop has VS 2019/MSVC 19.29 and bundled CMake, both outside PATH.
CUDA 12.6 Update 3 explicitly lists MSVC 192x/VS 2019 support in its
[archived requirements](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-microsoft-windows/index.html).
This is a host compiler compatibility reference, not a GPU/toolkit recommendation
for an unidentified lab GPU. Match the lab's GPU, OS, driver, and compiler to its
chosen toolkit. Refer to [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and that release's notes for exact driver requirements. The CUDA version displayed
by `nvidia-smi` is driver capability; it does not establish that `nvcc` is installed.

## Build and run: Windows PowerShell

From the copied project folder, with VS 2022 and CUDA integration installed:

```powershell
nvidia-smi
nvcc --version
cmake --version
cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64
cmake --build build-windows --config Release --parallel
.\build-windows\Release\vector_add.exe 1000
.\build-windows\Release\vector_add.exe 257
```

For a compatible VS 2019 toolchain use `-G "Visual Studio 16 2019"` instead,
with a different new build directory. Stop at any failed command and resolve it
before continuing. Use a new build directory when changing generator or toolkit.

## Build and run: Linux lab shell

Use a GPU node/session if the lab requires a scheduler. Load the lab's CUDA and
compiler modules first, following the administrator's instructions. If tools are
missing, ask the lab administrator to install the GPU driver, a supported toolkit,
GCC/G++, CMake, and a build tool using the
[official Linux guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

```bash
nvidia-smi
nvcc --version
g++ --version
cmake --version
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux --parallel
./build-linux/vector_add 1000
./build-linux/vector_add 257
```

## Machine-specific configuration

No toolkit path or GPU architecture is hardcoded. The default uses CMake's
detected CUDA compiler architecture default. For an explicit target, append
`-DCMAKE_CUDA_ARCHITECTURES=XX` to configure, replacing `XX` with the target GPU's
compute capability without the decimal point, supported by the chosen toolkit.
Use `"-DCMAKE_CUDA_ARCHITECTURES=XX;YY"` for multiple targets. CMake 3.24+ also
supports `-DCMAKE_CUDA_ARCHITECTURES=native` when configuring on the target GPU
machine. See [CMake architecture settings](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html).

With Ninja/Makefile generators, select a non-PATH toolkit using
`"-DCMAKE_CUDA_COMPILER=/absolute/path/to/nvcc"` and, if necessary,
`"-DCMAKE_CUDA_HOST_COMPILER=/absolute/path/to/supported/compiler"` at the first
configure. Use paths appropriate to the OS. For a Visual Studio generator,
select installed CUDA integration with `-T cuda=X.Y` or `-T cuda="<toolkit-path>"`
instead; CUDA host compiler selection is handled by the VS toolset.

## Expected GPU smoke-test output

```text
PASS: 1000 sums verified using 4 blocks of 256 threads.
PASS: 257 sums verified using 2 blocks of 256 threads.
```

Each run checks every sum against the CPU result and returns zero on success.
CUDA errors or mismatches print FAIL and return nonzero. Phase 2 is verified only
after the CMake build succeeds and these runs pass on a CUDA-capable GPU.

For a CPU indexing check only (no CUDA compilation or GPU execution), PowerShell:

```powershell
& ([scriptblock]::Create((Get-Content -LiteralPath .\check_indexing.ps1 -Raw)))
```

Later tool availability checks: `python --version` (Linux: `python3 --version`)
and `ncu --version`. No Python packages or profiling configuration are needed yet.
