# Phase 8 verification record

Checked 2026-11-06 on Windows 11/PowerShell. Phase 8 is **NOT COMPLETE**:
no CUDA-capable GPU measurements were collected, and no measured-result charts
were generated. Local code, CSV, and synthetic plotting checks passed.

## Inspection and environment

- No applicable AGENTS.md found in the project or ancestor directories.
- `git status --short` returned `fatal: not a git repository (or any of the parent directories): .git`.
- Existing sort implementations, generators, checker, and tests were preserved.
- Existing records identify Intel Iris Xe with no NVIDIA GPU; current PATH
  lookup found neither nvcc nor nvidia-smi.
- MSVC 19.29.30159.0, SDK 10.0.19041.0, CMake 3.20.21032501-MSVC_2.
- Python 3.12.10 and Matplotlib 3.8.1 available; no packages installed.
- No installation, system settings, or kernel optimization performed.

## Implementation

Extended the Phase 7 runner with pinned input/output host allocations, separate
same-stream CUDA event H2D/sort/D2H intervals, and steady-clock end-to-end wall
time. It requires Release, defaults to 3 warmups/10 trials, restores the same
keys per trial, verifies every output, exports per-trial and summary CSVs, and
reports failed/skipped cases. All required per-sort GPU work remains inside the
sort interval; allocations/workspace setup/CPU verification remain outside.
CPU-buildable CSV export calculates medians, throughput, and matching CUB ratios.

Added a Python matrix orchestrator with configurable sizes/distributions/seeds/
algorithms/counts, an all-seven smoke gate, logs/environment metadata, new-only
output directories, aggregated CSVs, and status records when blocked. Added
validation-first plotting and workload-specific performance/transfer summaries.
Fixture data and charts are kept under tests/fixtures and labelled synthetic.

## Actual CPU commands and results

From the project folder, using full executable paths (variable shown for brevity):

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
python -W ignore::DeprecationWarning tests\test_benchmark_pipeline.py
python -m py_compile scripts\run_benchmarks.py scripts\plot_benchmarks.py tests\test_benchmark_pipeline.py
```

Configure/build succeeded. Actual final CTest output:

```text
1/5 Test #1: cpu_correctness ..................   Passed    0.52 sec
2/5 Test #2: bitonic_cpu_logic ................   Passed    0.56 sec
3/5 Test #3: bitonic_shared_cpu_logic .........   Passed    0.82 sec
4/5 Test #4: radix_cpu_logic ..................   Passed    0.65 sec
5/5 Test #5: benchmark_options ................   Passed    0.46 sec

100% tests passed, 0 tests failed out of 5
Total Test time (real) =   3.05 sec
```

Python: **4 tests passed**. Coverage includes fixture charts, raw/summary median
and CUB consistency, rejection of corrupted/missing trial data, failed-case
filtering, and smoke-gate blocking. Python syntax compilation passed.

Additional actual fixture commands:

```powershell
New-Item -ItemType Directory -Force tests\fixtures\phase8_cpp_export | Out-Null
.\build-cpu\Release\benchmark_options_tests.exe tests\fixtures\phase8_cpp_export\case
python tests\test_benchmark_pipeline.py --write-fixture tests\fixtures\phase8_synthetic
python scripts\plot_benchmarks.py --input tests\fixtures\phase8_synthetic --output tests\fixtures\phase8_synthetic\charts --fixture
```

The generated C++ CSVs were copied to raw.csv/summary.csv in the same fixture
directory and read through Python's plot validator; two passed rows and the
synthetic failed/skipped rows, CSV escaping, medians, throughput, and a known
50%-of-CUB ratio were checked. Actual check output:

```text
PASS: C++ CSV export round-trip, failed/skipped rows, medians, throughput, and CUB percentage validated in Python.
PASS: 112 matrix cases recorded SKIPPED with reasons; zero measured trials.
```

Four fixture PNGs were generated: throughput, radix widths, and event breakdown
at 1024 and 65536. Throughput and stacked figures were visually inspected.
These are test artifacts, **not observations about sorting performance**.

## Actual CUDA blocker

Fresh configure attempt:

```powershell
& "$cmakeBin\cmake.exe" -S . -B build-phase8-cuda -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DBUILD_TESTING=ON
```

Exit code 1. First failing CUDA build-pipeline command:

```text
CMake Error ...
  No CUDA toolset found.
Call Stack ...
  CMakeLists.txt:9 (enable_language)
-- Configuring incomplete, errors occurred!
```

CUDA/CUB compilation, GPU correctness smoke, GPU event timing, and wall-clock
GPU trials remain unverified. This is toolchain discovery failure, before
source compilation or GPU execution.

Actual smoke-gated matrix invocation:

```powershell
python scripts\run_benchmarks.py --runner build-phase8-cuda\Release\sort_benchmark.exe --output results\phase8_local_blocked
```

Exit code 1; actual output:

```text
All-seven correctness smoke did not pass; no matrix measurements attempted
See results\phase8_local_blocked\smoke.log
```

The executable was unavailable because CUDA configuration failed. The full
matrix was never started. Its 112 algorithm cases are explicitly SKIPPED with
reasons and blank metrics in results/phase8_local_blocked/summary.csv. raw.csv
contains only its header. metadata.json records commands, failed environment
queries, OS, requested matrix, and blocked state; smoke.log records the missing
executable. These files contain no synthetic timings.

## Artifact locations

- Local blocked status: results/phase8_local_blocked/{metadata.json,smoke.log,raw.csv,summary.csv}.
- Synthetic plotting fixtures: tests/fixtures/phase8_synthetic/{metadata.json,raw.csv,summary.csv,charts/}.
- Synthetic C++ exporter fixtures: tests/fixtures/phase8_cpp_export/.
- Measured CSVs/charts: **pending**. README commands use results/phase8_lab/ and
  results/phase8_lab/charts/ on a CUDA-capable NVIDIA machine.

No fastest custom algorithm, measured CUB percentage, or size-dependent transfer
conclusion can be reported from this laptop. The plot script writes those
workload-specific findings to analysis.txt only from validated successful rows.

## Files changed

- src/sort_benchmark.cu; new src/benchmark_results.cpp.
- include/benchmark_options.hpp; new include/benchmark_results.hpp.
- CMakeLists.txt; tests/benchmark_options_tests.cpp.
- New scripts/run_benchmarks.py, scripts/plot_benchmarks.py, scripts/requirements.txt.
- New tests/test_benchmark_pipeline.py and labelled tests/fixtures/phase8_* artifacts.
- README.md, this PHASE8_RESULTS.md, and the local blocked-run status artifacts.

Next action: use README's Phase 8 Release build/test commands on a CUDA-capable
NVIDIA lab machine, run the smoke-gated default matrix, then plot its measured
CSVs. Phase 8 completes only after all required hardware measurements and their
charts exist. No Phase 9 profiling or kernel optimization was started.
