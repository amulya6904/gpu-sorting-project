# Phase 5 verification record

Checked 2026-11-06 on Windows 11/PowerShell with MSVC 19.29.30159.0,
Windows SDK 10.0.19041.0, and CMake 3.20.21032501-MSVC_2.

## Inspection and implementation

No applicable AGENTS.md found in the project or ancestor directories.
`git status --short` reported `fatal: not a git repository (or any of the parent
directories): .git`. Existing dataset/checker code, tests, vector_add.cu, and
historical results were preserved. Phase 2/4 diagnostics record Intel Iris Xe
with no NVIDIA GPU and no CUDA installation. The current PATH check
`Get-Command nvcc,nvidia-smi -ErrorAction SilentlyContinue` again found neither.
No installations or persistent settings changed.

Both entry points remain independently callable through bitonic_sort.cuh:
`bitonic_sort` (global) and `bitonic_sort_shared` (hybrid shared/global).
They reuse one internal global-step launcher and the existing uint32 dataset
generator/std::sort checker. Shared tiles are 256 keys, or n when n<256;
each block uses at most 256 threads and 1 KiB shared memory.

Initial tile-local stages use the global index to alternate ascending/descending
tile directions. Later stages launch global steps while stride >= tile size,
then finish all remaining strides in one shared-memory kernel per stage.
The current merge direction is preserved. Pair ownership is unique, and every
thread reaches load/step/store barriers. There are no partial tiles or early
returns within the kernels. A block-stride tile loop caps the grid at 65,535
blocks and has an additional barrier before reusing shared storage.

## Commands actually run

Using full tool paths from the project folder (shown with a variable for brevity):

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\cpu_tests.exe
.\build-cpu\Release\bitonic_logic_tests.exe
.\build-cpu\Release\bitonic_shared_logic_tests.exe
```

All CPU configure/build/test commands returned exit code 0. Actual CTest output:

```text
1/3 Test #1: cpu_correctness ..................   Passed    0.20 sec
2/3 Test #2: bitonic_cpu_logic ................   Passed    0.25 sec
3/3 Test #3: bitonic_shared_cpu_logic .........   Passed    0.94 sec

100% tests passed, 0 tests failed out of 3
Total Test time (real) =   1.41 sec
```

Actual executable output:

```text
PASS: 54 distribution/size cases, 66225 checks; generator properties and checker rejection tests passed.
PASS: CPU Bitonic logic check: 60 cases; unique pairs, step counts, full results, and invalid sizes verified. GPU NOT tested.
PASS: CPU shared Bitonic logic: 72 cases; full network equivalence, alternating tiles, pair ownership, routing, results, and rejection verified. GPU NOT tested.
```

Fresh CUDA configure attempt (new build directory):

```powershell
& "$cmakeBin\cmake.exe" -S . -B build-phase5-cuda -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=ON -DBUILD_TESTING=ON
```

Exit code 1; first blocker:

```text
CMake Error ...
  No CUDA toolset found.
Call Stack ...
  CMakeLists.txt:9 (enable_language)
-- Configuring incomplete, errors occurred!
```

This fails at CUDA toolchain discovery before compilation. CUDA compilation and
linking have not been verified. Neither GPU suite nor comparison executable could
be run. Diagnostic logs remain in the ignored build-phase5-cuda directory.

## Correctness and comparison status

| Check | Actual status |
| --- | --- |
| Phase 3 CPU tests | PASS |
| Phase 4 CPU logic | PASS |
| Shared schedule simulation | PASS: 72 cases, six distributions |
| CUDA compilation/linking | BLOCKED: missing CUDA toolset |
| Global GPU tests | PENDING |
| Shared GPU tests | PENDING; Phase 5 NOT GPU-verified |
| Global/shared median times and speedup | PENDING; no measured values |

The shared CPU/GPU test sizes are 0, 1, 2, 4, 8, 32, 128, 256, 512, 1024,
2048, and 65536. Invalid sizes 3, 31, and 257 are rejected. The GPU suite also
checks nullptr no-ops and nullptr rejection at n=2. CPU simulation verifies
schedule/partner logic, not CUDA compilation, barrier behavior, or GPU memory
execution. Actual GPU results are always compared in full against std::sort.

The prepared manual bitonic_compare executable uses random seed 12345 at sizes
256, 4096, and 65536, two warmups/version, and seven trials/version. It restores
the original data before each invocation and alternates version order. CUDA
events on the sort stream bracket the full sort-kernel sequence; allocation,
restoration, copies, and correctness checks are outside. Each warmup/trial is
verified, and median milliseconds and global/shared speedup are printed without
assuming shared memory wins. Timing may include GPU idle gaps during host launch
submission. This is a small comparison, not full benchmarking/profiling.

## Files changed

- CMakeLists.txt and README.md.
- include/bitonic_sort.cuh; new include/bitonic_shared_plan.hpp.
- src/bitonic_sort.cu (extract reusable global launcher); new
  src/bitonic_internal.cuh and src/bitonic_shared.cu.
- New tests/cuda_test_support.cuh, tests/bitonic_shared_logic_tests.cpp,
  tests/bitonic_shared_gpu_tests.cu, and tests/bitonic_compare.cu.
- This PHASE5_RESULTS.md.

Next action: copy the project excluding build directories to a CUDA-capable
NVIDIA lab machine and run README.md's configure/build/CTest commands, then
bitonic_compare. Retain both GPU correctness outputs and the actual timing table.
Phase 5 becomes GPU-verified only after shared correctness passes on hardware.
No Phase 6 Radix Sort work was started.
