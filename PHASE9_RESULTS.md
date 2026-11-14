# Phase 9 status and verification record

Checked 2026-11-06 on Windows 11/PowerShell. **Phase 9 remains incomplete**:
there are no real GPU profiler measurements, bottleneck findings, or optimization
claims. The discovery/capture workflow and reproducibility documentation are ready.

## Inspection

- No applicable AGENTS.md found in the project or ancestor directories.
- `git status --short`: `fatal: not a git repository (or any of the parent directories): .git`.
- Existing sort implementations, benchmark runner behavior, datasets, checker,
  correctness tests, and earlier result records were preserved.
- Phase 8's actual record is blocked_before_matrix, with 112 skipped cases and
  no measured trials. Synthetic plotting fixtures were not used for selection.
- Therefore the planned radix comparison uses Radix-1 and Radix-8, the extremes.
- ncu, nvcc, and nvidia-smi are absent from PATH. Standard NVIDIA Corporation
  and CUDA Toolkit installation directories are absent. Earlier Windows GPU
  detection recorded Intel Iris Xe without an NVIDIA GPU.
- Actual GPU/driver/toolkit/profiler versions are unavailable on this laptop.
  Installed CPU tools: MSVC 19.29.30159.0, SDK 10.0.19041.0, CMake
  3.20.21032501-MSVC_2, Python 3.12.10, Matplotlib 3.8.1.
- No system installation, counter permission, or GPU setting was changed.

## Files changed

- CMakeLists.txt: optional CUDA_LINEINFO adds -lineinfo to optimized CUDA builds.
- scripts/profile_kernels.py: discovery-first, exact-supported-metric selection,
  unprofiled correctness gate, one-launch capture filters, report import/parser,
  metric availability table, environment/commands/logs, and distinct error classes.
- tests/test_profile_workflow.py: independent stage/launch enumeration, CSV parser,
  error classification, and missing-profiler/no-measurement checks.
- README.md: exact Windows/Linux build/test/profile commands, capture mapping,
  supported-metric discovery, interpretation limits, and evidence-gated optimization.
- This PHASE9_RESULTS.md and profiles/phase9_local_* blocked discovery artifacts.

No profiling mode was added to the runner: kernel filters select a deterministic
launch within its first warmup while normal output verification finishes. Capture
commands never export Phase 8 timing CSVs. All timings printed under ncu are
explicitly excluded from benchmark comparisons.

## Commands actually run

From the project folder, using the installed CMake executable's full path:

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
python tests\test_profile_workflow.py
python -W ignore::DeprecationWarning tests\test_benchmark_pipeline.py
python -m py_compile scripts\profile_kernels.py tests\test_profile_workflow.py
```

CPU configure/build succeeded. Actual CTest output:

```text
1/5 Test #1: cpu_correctness ..................   Passed    0.23 sec
2/5 Test #2: bitonic_cpu_logic ................   Passed    0.30 sec
3/5 Test #3: bitonic_shared_cpu_logic .........   Passed    0.50 sec
4/5 Test #4: radix_cpu_logic ..................   Passed    0.41 sec
5/5 Test #5: benchmark_options ................   Passed    0.18 sec

100% tests passed, 0 tests failed out of 5
Total Test time (real) =   1.66 sec
```

Python profiling workflow: **3 tests passed**. Existing benchmark/plot pipeline:
**4 tests passed**. Python syntax compilation passed. Tests use no synthetic
profiler findings; the parser test has one labelled test CSV value only.

Fresh CUDA configure attempt:

```powershell
& "$cmakeBin\cmake.exe" -S . -B build-phase9-cuda -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DCUDA_LINEINFO=ON -DBUILD_TESTING=ON
```

Exit code 1, first failing CUDA build-pipeline command:

```text
CMake Error ...
  No CUDA toolset found.
Call Stack ...
  CMakeLists.txt:13 (enable_language)
-- Configuring incomplete, errors occurred!
```

This is toolchain discovery failure, before source compilation or GPU execution.
CUDA source-line compilation and GPU correctness remain unverified.

Actual final workflow invocation:

```powershell
python scripts\profile_kernels.py --runner build-phase9-cuda\Release\sort_benchmark.exe --output profiles\phase9_local_checked
```

Exit code 1, output:

```text
blocked_discovery: 0 measured metric rows; profiles\phase9_local_checked
```

The workflow attempted nvidia-smi and nvcc version queries and ncu --version,
--help, --list-sections, and --query-metrics --query-metrics-mode all --devices 0.
The actual profiler log records `[WinError 2] The system cannot find the file
specified`. This is a **missing profiler**, not an observed performance-counter
permission denial. Counter access has not been tested. ERR_NVGPUCTRPERM is only
the documented future error class if it actually occurs on a lab machine.

## Planned representative captures

At n=65536, random keys, seed 12345, one launch per report:

| Algorithm | Kernel | Captured stage/pass | Status |
| --- | --- | --- | --- |
| Global Bitonic | bitonic_step | stage 65536, stride 32768 (matching skip 120) | PENDING |
| Global Bitonic | bitonic_step | stage 256, stride 128 (skip 28) | PENDING |
| Shared Bitonic | bitonic_step | stage 65536, stride 32768 (skip 28) | PENDING |
| Shared Bitonic | tile_steps | initial local stages 2..256 (skip 0) | PENDING |
| Shared Bitonic | tile_steps | final stage, local strides 128..1 (skip 8) | PENDING |
| Radix-1 | histogram_step / prefix_step / scatter_step | pass 0, shift 0, separate captures | PENDING |
| Radix-8 | histogram_step / prefix_step / scatter_step | pass 0, shift 0, separate captures | PENDING |

Optional duplicate-heavy input adds six radix captures. n=262144 is also supported
with independently tested launch-index calculations. Radix pass 0 examines the
same original input for both widths. Grouped shared kernels do more work per
launch than one global compare step; their per-launch duration is not a like-for-
like full-sort comparison.

Discovery must precede selecting metrics. Candidate families are DRAM throughput
and byte counts, global load sectors/requests, shared bank conflicts, average
thread instructions per warp instruction, and uniform branch-target percentage.
Only exact names reported by the installed profiler/GPU are selected. Counters
can be absent by architecture/version; absence is not a zero measurement.

DRAM byte/conflict counters are direct measurements when available. Transaction
counts proxy access behavior, and active-thread/uniform-target metrics proxy
thread/branch behavior. Pair ownership, predication, bounds, and variable loop
length can all reduce active lanes. Source-level potential costs are documented
as hypotheses, not measured bottlenecks. No claim of memory saturation, bank
conflicts, histogram contention, or branch divergence is made locally.

## Report locations and remaining work

- profiles/phase9_local_checked/: metadata.json, query logs, and summary.csv.
- summary.csv has 99 UNAVAILABLE metric rows (11 planned captures × 9 candidate
  metrics), blank values/units, and missing-executable reasons.
- There are **no .ncu-rep reports** or real hardware counter values here.
- profiles/phase9_local_blocked/ preserves the initial discovery attempt.
- README lab commands use profiles/phase9_lab/ for future real reports, raw
  imported CSV logs, metadata, and the compact metric table.

No optimization was justified or applied. Before/after correctness, unprofiled
median sort timing, affected counter measurements, and repeatability assessment
are pending. Existing defaults remain unchanged. On hardware, choose at most
one small change only after evidence supports it; a speedup is optional.

Next action: follow README's Phase 9 Release+lineinfo build/test commands on a
CUDA-capable NVIDIA lab machine, query supported sections/metrics, and run the
capture script. If actual counter denial occurs, preserve the error and ask the
administrator to follow NVIDIA's permission guidance. Analyze real reports
alongside regular unprofiled timings before considering an optimization.
Phase 9 completes only with measured memory/thread explanations. No Phase 10
report/submission packaging was started.
