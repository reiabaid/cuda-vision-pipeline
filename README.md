# CUDA Vision Pipeline

Hand-written CUDA kernels for image preprocessing: grayscale, Gaussian blur, Sobel edges, histogram,
bilinear resize, and a fused ML preprocessing kernel. Every kernel is verified against OpenCV and
benchmarked against the GPU's theoretical memory bandwidth, so each speed claim comes with evidence
for *why* it is as fast as it is.

**Highlights** (RTX 3050 Ti Laptop, 4 GB, 192 GB/s peak DRAM bandwidth, cold L2, median of 200 runs)

| Kernel | Result at 1920x1080 | Correctness |
|---|---|---|
| Grayscale, 4 pixels/thread | **95.9% of peak**, 1.30x over naive (profiler: DRAM 92-96% busy) | bit-exact vs OpenCV, all 2^24 colours |
| Histogram, shared-memory privatization | **36x** faster than global atomics (random data), 93% of peak at 8K | exact vs `cv::calcHist` |
| Fused ML preprocessing (resize + BGR->RGB + normalize + HWC->CHW) | **95.9% of peak**, 1.30x over the 2-kernel version | within 1e-3 of a CPU float reference |
| Bilinear resize (2x down) | 92-96% of peak | max 1 LSB vs `cv::resize` |
| Gaussian blur 5x5, shared-memory tiles | 1.8-2.0x over naive, **17% of ideal bandwidth** (instruction-bound, see below) | bit-exact vs `cv::GaussianBlur` |
| Sobel 3x3, shared-memory tiles | 21.5% of ideal bandwidth | bit-exact vs `cv::Sobel` |

End-to-end demo (`cvk_demo`: grayscale -> blur -> Sobel on a 1920x1080 image, verified bit-exact against OpenCV):
GPU kernels 0.33 ms vs 4.7 ms for 16-thread OpenCV on CPU (**14x**); including PCIe upload/download
the GPU pipeline is 1.8 ms (**2.6x**), because copies dominate once the kernels are this fast.

## What is in here

| Path | Contents |
|---|---|
| `include/cvk.h` | Public C++ API (device-pointer functions, one CUDA stream parameter each) |
| `src/kernels/` | `grayscale.cu`, `blur.cu`, `sobel.cu`, `histogram.cu`, `resize.cu` (+ fused preprocessing), `common.cuh` (shared tile loader, border reflection) |
| `tests/` | One exact-match test per kernel against OpenCV; edge cases include 1x1, 1-pixel-wide images, odd sizes, misaligned pointers, degenerate sizes |
| `bench/` | GPU-event timing harness and per-kernel sweeps over 6 image sizes (`results/*.csv`) |
| `demo/demo.cpp` | End-to-end pipeline with verification and CPU comparison |
| `.github/workflows/` | CI (compile on every push; GPU test job on a self-hosted runner) and tagged releases |

## Methodology

- **Correctness first.** Kernels use integer fixed-point arithmetic and match OpenCV exactly
  (OpenCV's grayscale constants and blur weights were recovered by probing it; grayscale is tested over all
  16.7M colours). Resize and fused preprocessing use float and state their measured tolerance instead.
- **Timing.** CUDA events around the kernel, 10 warm-up launches, 200 timed iterations, median/min/p95.
  The L2 cache is flushed with a *read* kernel before each iteration (a write-based flush leaves dirty
  lines that distort timings; this was found by comparing the benchmark with Nsight Compute).
- **Percent of peak** = ideal bytes moved / time / theoretical DRAM bandwidth (computed from the driver's
  memory clock and bus width). Kernels that re-read data are penalised, by design.
- **Profiling.** Nsight Compute explains each result; see the reasoning below.

## Findings

- **Grayscale:** the naive kernel reads 3 strided bytes per thread and used 10.7 of every 32 bytes fetched.
  Loading 12 contiguous bytes (three 32-bit words) per thread of 4 pixels fixed it: DRAM throughput went from
  74.7% to 91.9-95.8% and compute stayed at 17%, so the kernel is at the memory ceiling.
- **Fusion:** doing resize, channel reorder, normalization and layout change in one kernel avoids an
  intermediate image and is 1.25-1.30x faster than two kernels (and slightly more precise, since there is no
  intermediate uint8 rounding).
- **Histogram:** privatizing bins in shared memory removes the contention of one global atomic per pixel.
  All-equal pixels (worst case for global atomics) widen the gap to ~60-98x; random data gives 36-48x.
- **Blur and Sobel are *not* memory-bound.** The tiled blur has DRAM at 15% busy but issue slots 58% busy and
  ~18.6 cycles between instructions. Removing integer division and border logic did not help (measured), so
  the remaining lever is more work per thread (several outputs per thread with word-sized shared/global
  accesses); this is the main open performance item.

## Build and run

Requires CMake >= 3.24, CUDA 12.x, a C++17 compiler (MSVC on Windows) and OpenCV >= 4.x with `core imgproc imgcodecs`
(tests and demo only; `-DCVK_BUILD_TESTS=OFF` builds the library and benchmarks without OpenCV).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # defaults to sm_86; override with -DCMAKE_CUDA_ARCHITECTURES=...
cmake --build build
ctest --test-dir build --output-on-failure                # 6 correctness suites
./build/cvk_demo [image.png] --out results/demo            # end-to-end pipeline
./build/bench_grayscale                                    # also: blur sobel histogram resize preprocess  (--warm for warm L2)
```

On Windows this project was developed with a fully project-local toolchain (no system installs); `scripts/env.ps1`
activates it for the current shell.

## Limitations

- Developed and measured on one laptop GPU. Absolute numbers drift with the laptop's power/thermal state
  (grayscale reached 54% of peak in one session and 74% in another with identical code), so speedups are
  quoted within a single run.
- Images must be contiguous (no row pitch); blur is single-channel 5x5 (sigma 1); exactness is verified
  against OpenCV 4.14.0 specifically.
- Resize uses float weights and differs from OpenCV's fixed-point by at most 1 LSB (up to ~13% of values for
  non-integer scale ratios).
- CI compiles the code on hosted runners (no GPU there); tests/benchmarks run in a self-hosted GPU job.

## Roadmap

Multi-output-per-thread blur and Sobel, separable filter variants, multi-channel blur, pitched images,
a streaming experiment for data larger than VRAM, and a TensorRT integration that feeds the fused
preprocessing kernel.
