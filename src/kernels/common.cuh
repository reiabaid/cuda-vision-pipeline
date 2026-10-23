// Device helpers shared by the stencil kernels (blur, Sobel). Internal, not installed.
#pragma once
#include <cuda_runtime.h>

namespace cvk_detail {

// BORDER_REFLECT_101: index -1 reads 1, index n reads n-2 (edge pixel not repeated).
// The loop keeps it correct even when the image is narrower than the stencil radius.
__device__ __forceinline__ int reflect101(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

// Cooperatively loads the BX x BY output tile at (x0, y0) plus an R-pixel halo into
// shared memory. Reflection is applied only for blocks whose halo leaves the image.
template <int BX, int BY, int R>
__device__ __forceinline__ void load_tile(unsigned char (&tile)[BY + 2 * R][BX + 2 * R],
                                          const unsigned char* __restrict__ src, int width,
                                          int height, int x0, int y0) {
  const bool interior = x0 >= R && y0 >= R && x0 + BX + R <= width && y0 + BY + R <= height;
  for (int ty = threadIdx.y; ty < BY + 2 * R; ty += BY)
    for (int tx = threadIdx.x; tx < BX + 2 * R; tx += BX) {
      int gy = y0 + ty - R, gx = x0 + tx - R;
      if (!interior) {
        gy = reflect101(gy, height);
        gx = reflect101(gx, width);
      }
      tile[ty][tx] = src[static_cast<size_t>(gy) * width + gx];
    }
}

}  // namespace cvk_detail
