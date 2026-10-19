#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                   cudaGetErrorString(err__), __FILE__, __LINE__);            \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// BGR (interleaved, uint8) -> gray (uint8). Matches cv::COLOR_BGR2GRAY.
// Naive baseline: one thread per pixel, byte loads.
void grayscale_gpu(const unsigned char* d_bgr, unsigned char* d_gray,
                   int width, int height, cudaStream_t stream = 0);

// Same result, bit-identical to grayscale_gpu. Four pixels per thread via 32-bit
// loads/stores. Requires tightly packed rows (no pitch); falls back to the naive
// kernel if either pointer is not 4-byte aligned.
void grayscale_gpu_vec4(const unsigned char* d_bgr, unsigned char* d_gray,
                        int width, int height, cudaStream_t stream = 0);
