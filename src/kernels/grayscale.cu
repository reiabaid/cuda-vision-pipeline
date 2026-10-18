#include "cvk.h"

// Naive version: one thread per pixel. Reads 3 B, writes 1 B per pixel.
__global__ void grayscale_kernel(const unsigned char* __restrict__ bgr,
                                 unsigned char* __restrict__ gray,
                                 int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  int i = y * width + x;
  float b = bgr[3 * i], g = bgr[3 * i + 1], r = bgr[3 * i + 2];
  // Same coefficients OpenCV uses (0.114, 0.587, 0.299), rounded to nearest.
  gray[i] = static_cast<unsigned char>(0.114f * b + 0.587f * g + 0.299f * r + 0.5f);
}

void grayscale_gpu(const unsigned char* d_bgr, unsigned char* d_gray,
                   int width, int height, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  grayscale_kernel<<<grid, block, 0, stream>>>(d_bgr, d_gray, width, height);
  CUDA_CHECK(cudaGetLastError());
}
