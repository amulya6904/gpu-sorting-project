# Phase 3 CPU verification

Verified 2026-11-06 on Windows 11 using PowerShell, MSVC 19.29.30159.0,
Windows SDK 10.0.19041.0, and CMake 3.20.21032501-MSVC_2.

No applicable AGENTS.md found. `git status --short` reported
`fatal: not a git repository (or any of the parent directories): .git`.
The existing vector_add.cu, check_indexing.ps1, and Phase 2 record were preserved.
No GPU tools were needed or installed.

## Actual commands

From `C:\Users\amuly\Desktop\GPU sorting`, using a new `build-cpu` directory:

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build-cpu -G 'Visual Studio 16 2019' -A x64 -DENABLE_CUDA=OFF -DBUILD_TESTING=ON
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build-cpu --config Release --parallel
& 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe' --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\Release\cpu_tests.exe
```

All four commands returned exit code 0. Configuration and generation succeeded;
the build produced `sorting_support.lib` and `cpu_tests.exe`.

Actual CTest output:

```text
    Start 1: cpu_correctness
1/1 Test #1: cpu_correctness ..................   Passed    0.49 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.50 sec
```

Actual direct executable output:

```text
PASS: 54 distribution/size cases, 66225 checks; generator properties and checker rejection tests passed.
```

## Coverage

- Six distributions at sizes 0, 1, 2, 4, 8, 16, 31, 257, and 65,536.
- Requested sizes, same-seed repeatability, different-seed effects on random
  distributions, and a known mt19937 output sample.
- Ascending/descending order and preservation of the random sample multiset;
  all-equal values; duplicate-heavy range and exact counts; boundary patterns
  including unsigned extrema.
- Acceptance of std::sort output, empty output, and singleton output.
- Rejection of unsorted data, missing/extra elements, sorted wrong values,
  incorrect duplicate counts, wrong singleton values, and reversed unsigned extrema.
- Diagnostic checks for the first mismatch index and size mismatch.

Phase 3 is complete. The default ENABLE_CUDA=ON path still enables CUDA and
builds vector_add from the unchanged source. Actual CUDA configuration/build/run
remain pending as recorded in SETUP_STATUS.md; they were not claimed verified
by the CPU tests. No sorting implementation or benchmark was added.
