#include "common.cuh"
#include "cvk.h"

namespace {

using cvk_detail::load_tile;
using cvk_detail::reflect101;

// OpenCV's bit-exact 8-bit Gaussian kernel for ksize=5, sigma=1: 8-bit fixed-point 1D
// weights (sum = 256), found by probing cv::GaussianBlur. Rows are NOT rounded between
// the horizontal and vertical passes; the only rounding is the final (sum + 2^15) >> 16.
constexpr int kRadius = 2;
__constant__ int c_w[2 * kRadius + 1] = {14, 62, 104, 62, 14};

// Naive version: one thread per output pixel, all 25 taps read from global memory.
__global__ void gaussian_blur5_naive_kernel(const unsigned char* __restrict__ src,
                                            unsigned char* __restrict__ dst, int width,
                                            int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  unsigned acc = 0;
#pragma unroll
  for (int dy = -kRadius; dy <= kRadius; ++dy) {
    const unsigned char* row = src + static_cast<size_t>(reflect101(y + dy, height)) * width;
#pragma unroll
    for (int dx = -kRadius; dx <= kRadius; ++dx)
      acc += c_w[dy + kRadius] * c_w[dx + kRadius] * row[reflect101(x + dx, width)];
  }
  dst[static_cast<size_t>(y) * width + x] = static_cast<unsigned char>((acc + (1u << 15)) >> 16);
}

// Tiled version: shared-memory tile + halo, horizontal pass into a 16-bit shared buffer
// (no rounding, as OpenCV), then the vertical pass.
constexpr int BX = 32, BY = 16, kHalo = 2 * kRadius;

__global__ void gaussian_blur5_tiled_kernel(const unsigned char* __restrict__ src,
                                            unsigned char* __restrict__ dst, int width,
                                            int height) {
  __shared__ unsigned char tile[BY + kHalo][BX + kHalo];
  __shared__ unsigned short horiz[BY + kHalo][BX];  // max 255 * 256 = 65280 fits in 16 bits

  load_tile<BX, BY, kRadius>(tile, src, width, height, blockIdx.x * BX, blockIdx.y * BY);
  __syncthreads();

  for (int r = threadIdx.y; r < BY + kHalo; r += BY) {
    unsigned acc = 0;
#pragma unroll
    for (int k = 0; k <= 2 * kRadius; ++k) acc += c_w[k] * tile[r][threadIdx.x + k];
    horiz[r][threadIdx.x] = static_cast<unsigned short>(acc);
  }
  __syncthreads();

  const int x = blockIdx.x * BX + threadIdx.x, y = blockIdx.y * BY + threadIdx.y;
  if (x >= width || y >= height) return;
  unsigned acc = 0;
#pragma unroll
  for (int k = 0; k <= 2 * kRadius; ++k) acc += c_w[k] * horiz[threadIdx.y + k][threadIdx.x];
  dst[static_cast<size_t>(y) * width + x] = static_cast<unsigned char>((acc + (1u << 15)) >> 16);
}

}  // namespace

void gaussian_blur5_naive(const unsigned char* d_src, unsigned char* d_dst, int width,
                          int height, cudaStream_t stream) {
  if (width <= 0 || height <= 0) return;
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  gaussian_blur5_naive_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height);
  CUDA_CHECK(cudaGetLastError());
}

void gaussian_blur5_tiled(const unsigned char* d_src, unsigned char* d_dst, int width,
                          int height, cudaStream_t stream) {
  if (width <= 0 || height <= 0) return;
  dim3 block(BX, BY);
  dim3 grid((width + BX - 1) / BX, (height + BY - 1) / BY);
  gaussian_blur5_tiled_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height);
  CUDA_CHECK(cudaGetLastError());
}
