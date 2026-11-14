# Phase 6 verification record

Checked 2026-11-06 on Windows 11/PowerShell using MSVC 19.29.30159.0,
Windows SDK 10.0.19041.0, and CMake 3.20.21032501-MSVC_2.

## Inspection

No applicable AGENTS.md found in the project or ancestor directories.
`git status --short` reported `fatal: not a git repository (or any of the parent
directories): .git`. Existing Phase 3 support code, both Bitonic implementations,
their tests, vector_add.cu, and historical verification records were preserved.
Phase 2/5 records identify Intel Iris Xe and no CUDA installation; the current
PATH check again found neither nvcc nor nvidia-smi. No installation or system
settings were changed.

## Implementation

One parameterized LSD implementation supports 1/2/4/8-bit digits, respectively
32/16/8/4 stable passes. Each pass queues exactly three kernels on the caller's
stream: tile histogram, exclusive digit/tile prefix, and stable scatter. The
scan is implemented here, without library primitives: one GPU thread per digit
loops over tiles, then sums lower digit totals to get its bucket base.
Histogram counts alone use atomics. Scatter ranks count earlier input indices
within the tile; no atomic scatter counters are used. Every required barrier
includes inactive threads in the final partial tile.

Caller-owned workspace contains an alternate key buffer and separate size_t
histogram/offset arrays. Allocation and copies are outside the sort, and buffers
swap roles after each pass. All full-sort pass counts are even, so the returned
final pointer equals the original keys pointer. Workspace requirements, pointer
checks, width validation, and no-op semantics are documented in radix_sort.cuh
and README.md. Callers must wait on the stream to surface execution errors.

## Actual commands

From the project folder, invoking the installed tools by full path:

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\radix_logic_tests.exe
```

CPU configure, build, CTest, and the direct radix test all returned exit code 0.
Actual output:

```text
1/4 Test #1: cpu_correctness ..................   Passed    0.21 sec
2/4 Test #2: bitonic_cpu_logic ................   Passed    0.26 sec
3/4 Test #3: bitonic_shared_cpu_logic .........   Passed    0.86 sec
4/4 Test #4: radix_cpu_logic ..................   Passed    0.41 sec

100% tests passed, 0 tests failed out of 4
Total Test time (real) =   1.78 sec

PASS: CPU radix logic: 240 full-sort cases, 16 stable-pass cases; scatter destinations, widths, shifts, workspace, and pass counts verified. GPU NOT tested.
```

Fresh CUDA configure attempt:

```powershell
& "$cmakeBin\cmake.exe" -S . -B build-phase6-cuda -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=ON -DBUILD_TESTING=ON
```

Exit code 1. First failing CUDA build-pipeline command; actual key error:

```text
CMake Error ...
  No CUDA toolset found.
Call Stack ...
  CMakeLists.txt:9 (enable_language)
-- Configuring incomplete, errors occurred!
```

This is missing CUDA toolchain discovery, before source compilation. It is not
a kernel compile error or GPU runtime failure. Logs remain in the ignored
build-phase6-cuda directory.

## Separate verification states

| Check | Actual status |
| --- | --- |
| Existing CPU support and both Bitonic logic suites | PASS |
| Radix CPU model, all four widths | PASS: 240 full-sort cases |
| Radix CPU stable-pass tests | PASS: 16 cases |
| CUDA compilation and linking | BLOCKED: no CUDA toolset |
| Actual GPU tests, including existing Bitonic suites | PENDING |
| GPU radix correctness/stability | PENDING; Phase 6 NOT GPU-verified |

Full-sort cases cover all six seeded distributions at n=0,1,2,31,255,256,257,
511,513,65536 for each width and compare the complete output with the Phase 3
std::sort reference. CPU checks additionally verify destination bounds and
unique scatter ownership, widths/shifts, workspace sizing/overflow, and pass
counts/buffer parity. The GPU suite checks no-ops, invalid widths, workspace
capacity rejection, and final buffer pointers.

The 16 targeted stable-pass cases use 769 distinguishable keys over three full
tiles and one partial tile, at low and high digit positions for every width.
Both mixed digits and all-same digits are tested. Every output key is compared
with a CPU std::stable_sort reference by the current digit. This exposes ordering
changes hidden by final comparisons of identical key values. The CPU model uses
serial input-order rank accumulation; GPU ranks explicitly count earlier local
indices. CPU success does not validate CUDA compilation, barriers, or hardware
execution.

## Files changed

- Updated CMakeLists.txt and README.md; existing CPU-only and CUDA targets remain.
- Added include/radix_plan.hpp and include/radix_sort.cuh.
- Added src/radix_sort.cu.
- Added tests/radix_test_cases.hpp, tests/radix_logic_tests.cpp, and
  tests/radix_gpu_tests.cu.
- Added this PHASE6_RESULTS.md.

Next action: copy the project excluding build directories to a CUDA-capable
NVIDIA lab machine and follow README.md's Windows/Linux configure, build, CTest,
and radix_gpu_tests commands. Keep the actual PASS output. Phase 6 is GPU-verified
only after all four widths and stable-scatter tests pass on hardware. No Phase 7
baseline integration or full benchmarking was started.
