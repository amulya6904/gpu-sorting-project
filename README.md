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
| CUDA build | Passed. CUDA 12.9 with MSVC 19.43 (VS 2022 Build Tools), `-DCMAKE_CUDA_ARCHITECTURES=61`. |
| GPU correctness | Passed. All 10 CTest checks passed on a GeForce GTX 1060 6GB, including the all-seven GPU smoke test and the Bitonic and Radix GPU correctness tests. |
| GPU benchmarks | Passed. The full matrix of 112 cases passed full-output verification on every warmup and trial. Results: `results/gtx1060-full-matrix-2026-11-19/`. |
| Profiling | Pending. Nsight Compute profiling has not been run yet. |

### Test Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1060 6GB (Pascal, compute capability 6.1) |
| Driver | 582.66 (supports CUDA 13.0) |
| CUDA Toolkit | 12.9.86. CUDA 13.x does not support Pascal GPUs, so 12.9 is required for this card. |
| Host compiler | MSVC 19.43.34808, Visual Studio 17 2022 generator, x64 |
| Build | Release |

The CUDA build on this machine used these configure flags:

```powershell
cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -T cuda="$env:CUDA_PATH" -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
```

## Results

Full matrix on the GTX 1060: 4 sizes × 4 distributions × 7 configurations, seed 12345, 3 warmups and 10 measured trials per case. The tables report median sort-only time in milliseconds, measured with CUDA events. They exclude host-to-device and device-to-host transfers.

### Random Keys

| Keys | Bitonic global | Bitonic shared | Radix-1 | Radix-2 | Radix-4 | Radix-8 | CUB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.349 | 0.064 | 0.626 | 0.316 | 0.165 | 0.094 | 0.037 |
| 65,536 | 0.990 | 0.398 | 2.998 | 1.544 | 0.869 | 1.250 | 0.431 |
| 262,144 | 1.880 | 1.265 | 11.117 | 5.507 | 3.128 | 4.713 | 0.598 |
| 1,048,576 | 10.890 | 6.680 | 42.281 | 18.969 | 10.994 | 15.499 | 0.471 |

### 1,048,576 Keys by Distribution

| Distribution | Bitonic global | Bitonic shared | Radix-1 | Radix-2 | Radix-4 | Radix-8 | CUB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 10.890 | 6.680 | 42.281 | 18.969 | 10.994 | 15.499 | 0.471 |
| Sorted | 10.089 | 5.971 | 37.710 | 18.969 | 10.954 | 15.259 | 0.450 |
| Reverse-sorted | 10.091 | 6.212 | 37.683 | 18.952 | 10.966 | 15.347 | 0.445 |
| Duplicate-heavy | 9.721 | 5.554 | 38.510 | 19.594 | 11.276 | 15.373 | 0.450 |

### Observations

- **Shared memory helps Bitonic.** Shared-memory Bitonic is 1.5–1.8× faster than global-memory Bitonic at 1,048,576 keys, and 5–8× faster at 1,024 keys.
- **Radix digit width.** For 65,536 keys and above, Radix-4 is the fastest radix configuration. Sort time roughly halves as the digit width doubles from 1 to 2 to 4 bits. Radix-8 is slower than Radix-4 at these sizes, probably because of the larger 256-bucket histogram and scatter cost. At 1,024 keys, Radix-8 is the fastest radix configuration.
- **CUB.** At 1,048,576 random keys, CUB sorts about 2.23 billion keys per second. Shared-memory Bitonic reaches 7.0% of CUB's throughput, Radix-4 4.3%, and Radix-1 1.1%.
- **Transfers.** At 1,048,576 keys, each host-to-device or device-to-host copy takes about 0.4 ms with pinned memory. Measured end to end, CUB takes 1.28 ms in total, so transfers take longer than its sort.
- **Distribution sensitivity.** Every configuration varies by about 10% or less across distributions. This is expected, because Bitonic and LSD radix sort do the same work regardless of input order.

`summary.csv` holds the medians, throughput, and percentage of CUB for every case. `raw.csv` holds every trial, and `metadata.json` records the GPU, driver, toolkit, and the exact commands.
