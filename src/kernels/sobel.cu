#include "common.cuh"
#include "cvk.h"

namespace {

constexpr int BX = 32, BY = 16;

// 3x3 Sobel, output = min(|gx| + |gy|, 255). Integer math, so it is exact.
__global__ void sobel_kernel(const unsigned char* __restrict__ src,
                             unsigned char* __restrict__ dst, int width, int height) {
  __shared__ unsigned char tile[BY + 2][BX + 2];
  cvk_detail::load_tile<BX, BY, 1>(tile, src, width, height, blockIdx.x * BX, blockIdx.y * BY);
  __syncthreads();

  const int x = blockIdx.x * BX + threadIdx.x, y = blockIdx.y * BY + threadIdx.y;
  if (x >= width || y >= height) return;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int p00 = tile[ty][tx], p01 = tile[ty][tx + 1], p02 = tile[ty][tx + 2];
  const int p10 = tile[ty + 1][tx], p12 = tile[ty + 1][tx + 2];
  const int p20 = tile[ty + 2][tx], p21 = tile[ty + 2][tx + 1], p22 = tile[ty + 2][tx + 2];
  const int gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
  const int gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
  const int m = abs(gx) + abs(gy);
  dst[static_cast<size_t>(y) * width + x] = static_cast<unsigned char>(m > 255 ? 255 : m);
}

}  // namespace

void sobel_magnitude(const unsigned char* d_gray, unsigned char* d_out, int width, int height,
                     cudaStream_t stream) {
  if (width <= 0 || height <= 0) return;
  dim3 block(BX, BY);
  dim3 grid((width + BX - 1) / BX, (height + BY - 1) / BY);
  sobel_kernel<<<grid, block, 0, stream>>>(d_gray, d_out, width, height);
  CUDA_CHECK(cudaGetLastError());
}
