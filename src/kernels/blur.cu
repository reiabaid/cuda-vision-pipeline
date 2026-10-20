#include "cvk.h"

namespace {

// OpenCV's bit-exact 8-bit Gaussian kernel for ksize=5, sigma=1: the 1D weights are
// 8-bit fixed point (sum = 256). Found by probing cv::GaussianBlur, not derived.
// Rows are NOT rounded between the horizontal and vertical passes; the only rounding
// is the final (sum + 2^15) >> 16.
constexpr int kRadius = 2;
__constant__ int c_w[2 * kRadius + 1] = {14, 62, 104, 62, 14};

// BORDER_REFLECT_101: index -1 reads 1, index n reads n-2 (edge pixel not repeated).
// The loop makes it correct even when the image is narrower than the kernel radius.
__device__ __forceinline__ int reflect101(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

// Naive version: one thread per output pixel, all 25 taps read from global memory.
__global__ void gaussian_blur5_naive_kernel(const unsigned char* __restrict__ src,
                                            unsigned char* __restrict__ dst,
                                            int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  unsigned acc = 0;
#pragma unroll
  for (int dy = -kRadius; dy <= kRadius; ++dy) {
    const unsigned char* row = src + static_cast<size_t>(reflect101(y + dy, height)) * width;
#pragma unroll
    for (int dx = -kRadius; dx <= kRadius; ++dx) {
      acc += c_w[dy + kRadius] * c_w[dx + kRadius] * row[reflect101(x + dx, width)];
    }
  }
  dst[static_cast<size_t>(y) * width + x] = static_cast<unsigned char>((acc + (1u << 15)) >> 16);
}

}  // namespace

void gaussian_blur5_naive(const unsigned char* d_src, unsigned char* d_dst,
                          int width, int height, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  gaussian_blur5_naive_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height);
  CUDA_CHECK(cudaGetLastError());
}
