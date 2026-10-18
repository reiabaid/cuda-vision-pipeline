# cuda-vision-pipeline

Hand-written CUDA kernels for core image operations, optimized step by step,
validated against OpenCV, and analyzed against the GPU's memory-bandwidth
roofline (RTX 3050 Ti Laptop, sm_86, ~192 GB/s).

## Plan
1. Baseline kernels: grayscale, blur, Sobel, resize (validated vs OpenCV)
2. Optimization ladder: naive -> shared-memory tiling -> separable blur -> pinned memory + streams
3. Harris corners, NMS, parallel reduction, histogram
4. Fused resize + normalize + HWC->CHW kernel
5. Roofline analysis with Nsight counters; scaling across image sizes

## Build
Requires CUDA Toolkit, CMake >= 3.24, MSVC Build Tools, OpenCV.

    cmake -S . -B build
    cmake --build build --config Release
    ctest --test-dir build

## Results
_TBD_
