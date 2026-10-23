#include "cvk.h"

namespace {

// Bilinear sample at destination pixel (dx, dy) with OpenCV's half-pixel-centre mapping.
// Returns the three interpolated channel values (BGR order) as floats.
__device__ __forceinline__ void sample_bilinear(const unsigned char* __restrict__ src, int sw,
                                                int sh, int dx, int dy, float scale_x,
                                                float scale_y, float out[3]) {
  const float fx_src = fmaxf((dx + 0.5f) * scale_x - 0.5f, 0.f);
  const float fy_src = fmaxf((dy + 0.5f) * scale_y - 0.5f, 0.f);
  const int x0 = min(static_cast<int>(fx_src), sw - 1), y0 = min(static_cast<int>(fy_src), sh - 1);
  const int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
  const float fx = fx_src - x0, fy = fy_src - y0;

  const unsigned char* r0 = src + static_cast<size_t>(y0) * sw * 3;
  const unsigned char* r1 = src + static_cast<size_t>(y1) * sw * 3;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float top = r0[x0 * 3 + c] + fx * (r0[x1 * 3 + c] - r0[x0 * 3 + c]);
    const float bot = r1[x0 * 3 + c] + fx * (r1[x1 * 3 + c] - r1[x0 * 3 + c]);
    out[c] = top + fy * (bot - top);
  }
}

__global__ void resize_bgr_kernel(const unsigned char* __restrict__ src, int sw, int sh,
                                  unsigned char* __restrict__ dst, int dw, int dh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float v[3];
  sample_bilinear(src, sw, sh, x, y, static_cast<float>(sw) / dw, static_cast<float>(sh) / dh, v);
  unsigned char* o = dst + (static_cast<size_t>(y) * dw + x) * 3;
  for (int c = 0; c < 3; ++c) o[c] = static_cast<unsigned char>(v[c] + 0.5f);
}

// Fused: resize + BGR->RGB + (x/255 - mean) * inv_std + HWC->CHW float, in one pass with
// no intermediate image. The unfused path rounds to uint8 between the steps; this does not.
__global__ void preprocess_fused_kernel(const unsigned char* __restrict__ src, int sw, int sh,
                                        float* __restrict__ dst, int dw, int dh, NormParams p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float v[3];
  sample_bilinear(src, sw, sh, x, y, static_cast<float>(sw) / dw, static_cast<float>(sh) / dh, v);
  const size_t plane = static_cast<size_t>(dw) * dh, i = static_cast<size_t>(y) * dw + x;
#pragma unroll
  for (int c = 0; c < 3; ++c)  // output channel c is RGB; v is BGR
    dst[c * plane + i] = (v[2 - c] * (1.f / 255.f) - p.mean[c]) * p.inv_std[c];
}

// Second stage of the unfused path: uint8 HWC BGR -> normalized float CHW RGB.
__global__ void normalize_nchw_kernel(const unsigned char* __restrict__ in,
                                      float* __restrict__ dst, int w, int h, NormParams p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const size_t plane = static_cast<size_t>(w) * h, i = static_cast<size_t>(y) * w + x;
  const unsigned char* px = in + i * 3;
#pragma unroll
  for (int c = 0; c < 3; ++c) dst[c * plane + i] = (px[2 - c] * (1.f / 255.f) - p.mean[c]) * p.inv_std[c];
}

dim3 grid_for(int w, int h, dim3 b) { return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y); }

}  // namespace

void resize_bilinear_bgr(const unsigned char* d_src, int sw, int sh, unsigned char* d_dst,
                         int dw, int dh, cudaStream_t stream) {
  if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
  dim3 b(32, 8);
  resize_bgr_kernel<<<grid_for(dw, dh, b), b, 0, stream>>>(d_src, sw, sh, d_dst, dw, dh);
  CUDA_CHECK(cudaGetLastError());
}

void preprocess_nchw(const unsigned char* d_src, int sw, int sh, float* d_dst, int dw, int dh,
                     const NormParams& p, cudaStream_t stream) {
  if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
  dim3 b(32, 8);
  preprocess_fused_kernel<<<grid_for(dw, dh, b), b, 0, stream>>>(d_src, sw, sh, d_dst, dw, dh, p);
  CUDA_CHECK(cudaGetLastError());
}

void preprocess_nchw_unfused(const unsigned char* d_src, int sw, int sh, float* d_dst, int dw,
                             int dh, const NormParams& p, unsigned char* d_tmp,
                             cudaStream_t stream) {
  if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
  resize_bilinear_bgr(d_src, sw, sh, d_tmp, dw, dh, stream);
  dim3 b(32, 8);
  normalize_nchw_kernel<<<grid_for(dw, dh, b), b, 0, stream>>>(d_tmp, d_dst, dw, dh, p);
  CUDA_CHECK(cudaGetLastError());
}
