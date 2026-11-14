# CUDA GPU Sorting Benchmark

A CUDA C++ project for comparing parallel sorting algorithms on 32-bit unsigned integer keys. It evaluates how shared memory affects Bitonic Sort, how radix digit width affects throughput, and how six custom configurations compare with NVIDIA CUB on identical workloads.

## Sorting Configurations

| Configuration | Approach |
| --- | --- |
| Global-memory Bitonic | Runs each compare-and-swap step on keys stored in device global memory. |
| Shared-memory Bitonic | Performs tile-local steps in shared memory and uses global memory for comparisons across tiles. |
| Radix-1, Radix-2, Radix-4, Radix-8 | Use stable least-significant-digit radix sorting. The four digit widths require 32, 16, 8, and 4 passes, respectively. |
| NVIDIA CUB | Uses `cub::DeviceRadixSort::SortKeys` as the GPU performance baseline. |

The Bitonic configurations use power-of-two input sizes. Radix and CUB also support non-power-of-two inputs. Correctness is checked by comparing the full output with a CPU reference produced by `std::sort`.

## Build and Test

Run these commands from the directory containing `CMakeLists.txt`. The Windows commands assume a Visual Studio generator and an x64 Developer PowerShell.

### CPU-Only Build

Requires CMake and a C++ compiler. This build runs the CPU checks on machines without a CUDA toolchain.

```powershell
cmake -S . -B build-cpu -DENABLE_CUDA=OFF
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

### CUDA Build

Requires a CUDA-capable NVIDIA GPU for execution, an NVIDIA driver, the CUDA Toolkit with CUB headers, CMake, and a compatible host compiler. Kernel profiling also uses NVIDIA Nsight Compute.

```powershell
nvidia-smi
nvcc --version
cmake -S . -B build-cuda -DENABLE_CUDA=ON
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
```

The CUDA benchmark target is `sort_benchmark`. With the Visual Studio generator, the Release executable is `build-cuda\Release\sort_benchmark.exe`. Start with a smoke test of all seven configurations on 1,024 seeded random keys before collecting performance data.

## Benchmark Methodology

- **Workloads.** The benchmark matrix uses 1,024, 65,536, 262,144, and 1,048,576 keys across random, sorted, reverse-sorted, and duplicate-heavy distributions. Each algorithm receives the same seeded input for a given test case.
- **Timing.** Warm up each configuration, restore the original input before each measured trial, and report median timings. Use CUDA events to measure host-to-device transfer, GPU sorting, and device-to-host transfer separately. Measure end-to-end elapsed time with a wall clock.
- **Throughput.** Divide the key count by the median sort time in seconds. Compare each custom configuration with CUB using the same input size, distribution, seed, and GPU. Keep allocation and correctness checks outside the sort-only interval.
- **Profiling.** Use Nsight Compute to inspect memory throughput and warp behavior in representative Bitonic and Radix kernels. Collect benchmark timings independently of profiler runs.

## Verification Status

| Check | Status |
| --- | --- |
| CPU Release build and tests | Passed. All five CPU checks passed on Windows. |
| CUDA build | Pending on a CUDA-equipped machine. Local CMake configuration reported `No CUDA toolset found`. |
| GPU correctness, benchmarks, and profiling | Pending validation on a CUDA-capable NVIDIA GPU. |

The development laptop detects Intel Iris Xe graphics. GPU verification and performance measurements require access to a CUDA-capable lab, teammate, or remote machine.
