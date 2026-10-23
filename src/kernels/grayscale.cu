#include <cstdint>

#include "cvk.h"

namespace {

// OpenCV 4.14's 15-bit fixed-point BGR2GRAY: 0.114, 0.587, 0.299 scaled by 2^15 (the
// integers sum to exactly 2^15), rounded to nearest. Verified identical to
// cv::cvtColor on all 2^24 colours. Integer math is exact, so every variant gives
// identical output (a float version let the compiler fuse multiply-adds differently
// per kernel and differ by 1 in rare cases).
__device__ __forceinline__ unsigned char to_gray(unsigned b, unsigned g, unsigned r) {
  return static_cast<unsigned char>((3735u * b + 19235u * g + 9798u * r + (1u << 14)) >> 15);
}

__device__ __forceinline__ unsigned byte_of(uint32_t word, int i) {
  return (word >> (8 * i)) & 0xffu;
}

// Naive version: one thread per pixel. Reads 3 B, writes 1 B per pixel.
__global__ void grayscale_kernel(const unsigned char* __restrict__ bgr,
                                 unsigned char* __restrict__ gray,
                                 int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const size_t i = static_cast<size_t>(y) * width + x;  // size_t: 3*i overflows int past ~715M px
  gray[i] = to_gray(bgr[3 * i], bgr[3 * i + 1], bgr[3 * i + 2]);
}

// Vectorized version: one thread per 4 pixels = 12 input bytes = three 32-bit
// words (little-endian):  w0 = B0 G0 R0 B1 | w1 = G1 R1 B2 G2 | w2 = R2 B3 G3 R3.
// The three loads are independent, so they are in flight together, and the four
// results leave as a single 32-bit store.
__global__ void grayscale_vec4_kernel(const uint32_t* __restrict__ in,
                                      uint32_t* __restrict__ out, size_t n_quads) {
  size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n_quads) return;

  const uint32_t w0 = in[3 * t], w1 = in[3 * t + 1], w2 = in[3 * t + 2];

  const uint32_t g0 = to_gray(byte_of(w0, 0), byte_of(w0, 1), byte_of(w0, 2));
  const uint32_t g1 = to_gray(byte_of(w0, 3), byte_of(w1, 0), byte_of(w1, 1));
  const uint32_t g2 = to_gray(byte_of(w1, 2), byte_of(w1, 3), byte_of(w2, 0));
  const uint32_t g3 = to_gray(byte_of(w2, 1), byte_of(w2, 2), byte_of(w2, 3));

  out[t] = g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
}

// Handles the last (n mod 4) pixels that do not fill a whole quad.
__global__ void grayscale_tail_kernel(const unsigned char* __restrict__ bgr,
                                      unsigned char* __restrict__ gray,
                                      size_t first, size_t n) {
  size_t i = first + threadIdx.x;
  if (i < n) gray[i] = to_gray(bgr[3 * i], bgr[3 * i + 1], bgr[3 * i + 2]);
}

}  // namespace

void grayscale_gpu(const unsigned char* d_bgr, unsigned char* d_gray,
                   int width, int height, cudaStream_t stream) {
  if (width <= 0 || height <= 0) return;
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  grayscale_kernel<<<grid, block, 0, stream>>>(d_bgr, d_gray, width, height);
  CUDA_CHECK(cudaGetLastError());
}

void grayscale_gpu_vec4(const unsigned char* d_bgr, unsigned char* d_gray,
                        int width, int height, cudaStream_t stream) {
  if (width <= 0 || height <= 0) return;
  // Word loads/stores need 4-byte alignment; cudaMalloc gives far more, but a
  // caller may pass an offset pointer. Fall back rather than fault.
  const bool aligned = reinterpret_cast<uintptr_t>(d_bgr) % 4 == 0 &&
                       reinterpret_cast<uintptr_t>(d_gray) % 4 == 0;
  if (!aligned) {
    grayscale_gpu(d_bgr, d_gray, width, height, stream);
    return;
  }

  // Pointwise op, so treat the image as a flat array (rows must be tightly packed).
  const size_t n = static_cast<size_t>(width) * height;
  const size_t quads = n / 4;
  if (quads > 0) {
    constexpr int kBlock = 256;
    const unsigned grid = static_cast<unsigned>((quads + kBlock - 1) / kBlock);
    grayscale_vec4_kernel<<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(d_bgr), reinterpret_cast<uint32_t*>(d_gray), quads);
    CUDA_CHECK(cudaGetLastError());
  }
  const size_t tail = n - quads * 4;
  if (tail > 0) {
    grayscale_tail_kernel<<<1, static_cast<unsigned>(tail), 0, stream>>>(d_bgr, d_gray,
                                                                          quads * 4, n);
    CUDA_CHECK(cudaGetLastError());
  }
}
