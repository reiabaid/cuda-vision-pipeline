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

// 5x5 Gaussian blur (sigma 1) of a single-channel 8-bit image, reflect-101 borders.
// Bit-exact with cv::GaussianBlur(src, dst, Size(5,5), 1.0, 1.0, BORDER_REFLECT_101)
// as of OpenCV 4.14. Rows must be tightly packed. src and dst must not alias.
// Naive baseline: one thread per pixel, 25 global loads each.
void gaussian_blur5_naive(const unsigned char* d_src, unsigned char* d_dst,
                          int width, int height, cudaStream_t stream = 0);

// Same result, bit-identical to gaussian_blur5_naive. Shared-memory tiles with halo,
// separable passes with a 16-bit intermediate.
void gaussian_blur5_tiled(const unsigned char* d_src, unsigned char* d_dst,
                          int width, int height, cudaStream_t stream = 0);

// 3x3 Sobel edge magnitude of a single-channel 8-bit image: min(|gx| + |gy|, 255),
// reflect-101 borders. Bit-exact with cv::Sobel (CV_16S) + saturating |.|+|.|.
void sobel_magnitude(const unsigned char* d_gray, unsigned char* d_out,
                     int width, int height, cudaStream_t stream = 0);

// 256-bin histogram of a single-channel 8-bit image. d_hist holds 256 unsigned ints and is
// zeroed by the call. Exact. The shared-memory version privatizes bins per block;
// the naive version does one global atomic per pixel.
void histogram256(const unsigned char* d_gray, unsigned* d_hist,
                  int width, int height, cudaStream_t stream = 0);
void histogram256_naive(const unsigned char* d_gray, unsigned* d_hist,
                        int width, int height, cudaStream_t stream = 0);

// Bilinear resize of a packed 8-bit BGR image (half-pixel centres, as cv::INTER_LINEAR).
// Within 1 LSB of OpenCV (OpenCV uses 11-bit fixed-point weights, this uses float).
void resize_bilinear_bgr(const unsigned char* d_src, int src_w, int src_h,
                         unsigned char* d_dst, int dst_w, int dst_h, cudaStream_t stream = 0);

// Normalization for ML inputs; channel order is RGB (index 0 = R).
struct NormParams {
  float mean[3];
  float inv_std[3];  // 1 / std
};

// Fused ML preprocessing: bilinear resize + BGR->RGB + (x/255 - mean) * inv_std +
// HWC->CHW float32, one kernel, no intermediate image. d_dst holds 3*dst_w*dst_h floats.
void preprocess_nchw(const unsigned char* d_src, int src_w, int src_h,
                     float* d_dst, int dst_w, int dst_h,
                     const NormParams& params, cudaStream_t stream = 0);

// Same result via two kernels and a uint8 intermediate (d_tmp: 3*dst_w*dst_h bytes), kept
// as the baseline for the fusion benchmark. Differs from the fused path by the intermediate
// rounding (up to 0.5/255/std per value).
void preprocess_nchw_unfused(const unsigned char* d_src, int src_w, int src_h,
                             float* d_dst, int dst_w, int dst_h, const NormParams& params,
                             unsigned char* d_tmp, cudaStream_t stream = 0);