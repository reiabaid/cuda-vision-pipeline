#include <cstdint>

#include "cvk.h"

namespace {

// Baseline: every pixel does an atomic add straight into the global histogram.
// Pixels with equal values collide on the same address and serialize.
__global__ void hist_global_kernel(const unsigned char* __restrict__ in, unsigned* hist,
                                   size_t n) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    atomicAdd(&hist[in[i]], 1u);
}

// Privatized: each block counts into a fast shared-memory histogram (atomics on shared
// memory are far cheaper), reading 4 pixels per 32-bit load, then merges 256 bins once.
__global__ void hist_shared_kernel(const unsigned char* __restrict__ in, unsigned* hist,
                                   size_t n) {
  __shared__ unsigned local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
  __syncthreads();

  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t* words = reinterpret_cast<const uint32_t*>(in);
  const size_t n_words = n / 4;
  for (size_t i = tid; i < n_words; i += stride) {
    const uint32_t w = words[i];
    atomicAdd(&local[w & 255u], 1u);
    atomicAdd(&local[(w >> 8) & 255u], 1u);
    atomicAdd(&local[(w >> 16) & 255u], 1u);
    atomicAdd(&local[w >> 24], 1u);
  }
  for (size_t i = n_words * 4 + tid; i < n; i += stride) atomicAdd(&local[in[i]], 1u);
  __syncthreads();

  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    if (local[i]) atomicAdd(&hist[i], local[i]);
}

constexpr int kBlock = 256;

unsigned grid_for(size_t n) {
  const size_t blocks = (n / 4 + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 1 ? 1 : (blocks > 1024 ? 1024 : blocks));
}

}  // namespace

// Zeroes the output histogram (when given) and returns the pixel count (0 for bad sizes).
static size_t zero_and_count(unsigned* d_hist, int width, int height, cudaStream_t stream) {
  if (d_hist) CUDA_CHECK(cudaMemsetAsync(d_hist, 0, 256 * sizeof(unsigned), stream));
  return static_cast<size_t>(width > 0 ? width : 0) * (height > 0 ? height : 0);
}

void histogram256_naive(const unsigned char* d_gray, unsigned* d_hist, int width, int height,
                        cudaStream_t stream) {
  const size_t n = zero_and_count(d_hist, width, height, stream);
  if (n == 0) return;
  hist_global_kernel<<<grid_for(n), kBlock, 0, stream>>>(d_gray, d_hist, n);
  CUDA_CHECK(cudaGetLastError());
}

void histogram256(const unsigned char* d_gray, unsigned* d_hist, int width, int height,
                  cudaStream_t stream) {
  if (reinterpret_cast<uintptr_t>(d_gray) % 4 != 0) {  // word loads need alignment
    histogram256_naive(d_gray, d_hist, width, height, stream);
    return;
  }
  const size_t n = zero_and_count(d_hist, width, height, stream);
  if (n == 0) return;
  hist_shared_kernel<<<grid_for(n), kBlock, 0, stream>>>(d_gray, d_hist, n);
  CUDA_CHECK(cudaGetLastError());
}