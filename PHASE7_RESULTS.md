# Phase 7 verification record

Checked 2026-11-06 on Windows 11/PowerShell using MSVC 19.29.30159.0 and
CMake 3.20.21032501-MSVC_2. No applicable AGENTS.md found. Git status reported
`fatal: not a git repository (or any of the parent directories): .git`.
Existing sorting implementations, generators, correctness checker, tests, and
historical records were preserved. Current PATH lookup found neither nvcc nor
nvidia-smi; earlier diagnostics identify Intel Iris Xe without an NVIDIA GPU.
No installation or persistent settings were changed.

## Changes

- Added src/sort_benchmark.cu: one runner for global/shared Bitonic, Radix-1/2/4/8,
  and CUB SortKeys, using original input restoration, CUDA-event sort-only timing,
  warmups, repeated trials, complete verification, and median reporting.
- Added include/benchmark_options.hpp: CUDA-independent options and size contracts.
- Added tests/benchmark_options_tests.cpp: CLI selections, distributions, numeric
  bounds, invalid inputs, and power-of-two support checks.
- Updated CMakeLists.txt: CUDA-only runner with optional toolkit CUB baseline;
  two GPU smoke tests plus the CPU option test. CPU-only builds never include CUB.
- Updated README.md with exact Windows/Linux commands and CUB fallback guidance.
- Added this verification record.

CUB's pointer overload receives separate input/output buffers and bits [0,32)
on the sorting stream. Temporary bytes are queried and allocated before warmups,
then reused. The actual explicit CUB output buffer is verified, while custom
full sorts use the original key buffer. All warmups and trials restore the same
generated input, and allocations/restoration/copies/reference checking are
outside the timed interval. Errors produce nonzero exit codes. This is a small
single-case runner, not Phase 8's full benchmark matrix.

## Actual commands

From the project folder, using full executable paths (shown with a variable):

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\benchmark_options_tests.exe
```

All returned exit code 0. Actual output:

```text
1/5 Test #1: cpu_correctness ..................   Passed    0.19 sec
2/5 Test #2: bitonic_cpu_logic ................   Passed    0.29 sec
3/5 Test #3: bitonic_shared_cpu_logic .........   Passed    0.73 sec
4/5 Test #4: radix_cpu_logic ..................   Passed    0.34 sec
5/5 Test #5: benchmark_options ................   Passed    0.18 sec

100% tests passed, 0 tests failed out of 5
Total Test time (real) =   1.76 sec

PASS: benchmark option parsing, selections, size contracts, and invalid arguments verified on CPU. GPU NOT tested.
```

Fresh CUDA/CUB configure attempt:

```powershell
& "$cmakeBin\cmake.exe" -S . -B build-phase7-cuda -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=ON -DENABLE_CUB_BASELINE=ON -DBUILD_TESTING=ON
```

Exit code 1; first failing build-pipeline command. Actual key error:

```text
CMake Error ...
  No CUDA toolset found.
Call Stack ...
  CMakeLists.txt:9 (enable_language)
-- Configuring incomplete, errors occurred!
```

This fails before CUDA source or CUB header compilation. No claim is made that
CUB is missing/incompatible independently of the absent CUDA toolchain. Actual
CUB header/API compatibility and CUDA compilation/linking remain unverified.

## Verification states

| Check | Actual status |
| --- | --- |
| Existing four CPU suites | PASS |
| Runner CPU options/contracts | PASS |
| CUDA/CUB compilation and linking | BLOCKED: no CUDA toolset |
| 1024-key random all-seven GPU smoke | PENDING |
| 257-key duplicate-heavy Radix/CUB GPU smoke | PENDING |
| Actual sort-only timings | PENDING; no values claimed |

No configuration has been newly GPU-verified. CTest registrations and lab
commands are ready, but hardware smoke outputs are not fabricated. The custom-
only CUB-disabled CUDA build also remains unverified because CUDA is absent.

Next action: copy the project excluding build directories to a CUDA-capable
NVIDIA lab machine, follow README.md's Phase 7 configure/build/test commands,
then retain direct runner output for both cases. Phase 7 becomes GPU-verified
only after all seven pass the same-input 1024-key case on hardware. No Phase 8
full matrix, transfer breakdown, throughput comparison, or charts were started.
