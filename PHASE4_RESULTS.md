# Phase 4 verification record

Checked 2026-11-06 on Windows 11/PowerShell using MSVC 19.29.30159.0 and
CMake 3.20.21032501-MSVC_2. No applicable AGENTS.md found. Git status reported
`fatal: not a git repository (or any of the parent directories): .git`.
Existing Phase 1 and Phase 3 source/tests were preserved.

Reviewed Phase 2's SETUP_STATUS.md: Windows detected Intel Iris Xe with no NVIDIA
GPU; CUDA tools were absent from PATH and standard installation locations.
`Get-Command nvcc,nvidia-smi -ErrorAction SilentlyContinue` again found neither.
No installations or persistent settings were changed.

## Commands actually run

The full CMake tool directory was used because it is not in ordinary PowerShell PATH:

```powershell
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
& "$cmakeBin\cmake.exe" -S . -B build-cpu -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& "$cmakeBin\cmake.exe" --build build-cpu --config Release --parallel
& "$cmakeBin\ctest.exe" --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\cpu_tests.exe
.\build-cpu\Release\bitonic_logic_tests.exe
```

CPU configure, build, CTest, and both executables returned exit code 0.
Actual output:

```text
1/2 Test #1: cpu_correctness ..................   Passed    0.32 sec
2/2 Test #2: bitonic_cpu_logic ................   Passed    0.93 sec
100% tests passed, 0 tests failed out of 2
Total Test time (real) =   1.33 sec

PASS: 54 distribution/size cases, 66225 checks; generator properties and checker rejection tests passed.
PASS: CPU Bitonic logic check: 60 cases; unique pairs, step counts, full results, and invalid sizes verified. GPU NOT tested.
```

Fresh CUDA configure attempt:

```powershell
& "$cmakeBin\cmake.exe" -S . -B build-phase4-cuda -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=ON -DBUILD_TESTING=ON
```

Exit code 1. First failing CUDA build-pipeline command; key actual error:

```text
CMake Error ...
  No CUDA toolset found.
Call Stack ...
  CMakeLists.txt:9 (enable_language)
-- Configuring incomplete, errors occurred!
```

This is missing CUDA toolchain discovery, before source compilation. It is not
a kernel compilation failure or GPU runtime error. CMake logs remain in the
ignored build-phase4-cuda directory.

## Separate verification states

- CPU support regression suite: PASS.
- CPU Bitonic schedule/partner simulation: PASS for six distributions and ten
  sizes including 0, 1, and 65,536. Each step checks partner validity, reciprocal
  pairing, exactly one owner per key, and expected total step count. Invalid
  sizes 3, 31, and 257 are rejected before any step.
- CUDA source compilation/linking: BLOCKED, unverified.
- Actual GPU correctness: PENDING, not GPU-verified.
- CUDA-event smoke measurement: PENDING; no timing value claimed.

The CUDA test driver contains 60 full-reference correctness cases, no-op and
257-rejection checks, plus one event-timed random 65,536-key sort. The timed
interval includes the full kernel sequence, excludes allocation and copies, and
is only a smoke measurement. Tests fail if no CUDA GPU is available.

Next action: copy this project excluding build directories to a CUDA-capable
NVIDIA machine, follow README.md's Windows or Linux commands, and retain the GPU
test PASS output and smoke measurement. Phase 4 is GPU-verified only when those
correctness tests pass on an actual CUDA-capable GPU. No Phase 5 work started.
