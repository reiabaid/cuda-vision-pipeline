#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cvk.h"

namespace {

const NormParams kImageNet = {{0.485f, 0.456f, 0.406f}, {1 / 0.229f, 1 / 0.224f, 1 / 0.225f}};

// CPU float reference with the same sampling as the kernel (half-pixel centres, clamped).
std::vector<float> cpu_reference(const cv::Mat& src, int dw, int dh, const NormParams& p) {
  const int sw = src.cols, sh = src.rows;
  std::vector<float> out(3 * static_cast<size_t>(dw) * dh);
  for (int y = 0; y < dh; ++y)
    for (int x = 0; x < dw; ++x) {
      const float fx = std::max((x + 0.5f) * sw / dw - 0.5f, 0.f), fy = std::max((y + 0.5f) * sh / dh - 0.5f, 0.f);
      const int x0 = std::min(static_cast<int>(fx), sw - 1), y0 = std::min(static_cast<int>(fy), sh - 1);
      const int x1 = std::min(x0 + 1, sw - 1), y1 = std::min(y0 + 1, sh - 1);
      const float ax = fx - x0, ay = fy - y0;
      for (int c = 0; c < 3; ++c) {  // c = RGB channel; source is BGR
        auto px = [&](int yy, int xx) { return static_cast<float>(src.at<cv::Vec3b>(yy, xx)[2 - c]); };
        const float top = px(y0, x0) + ax * (px(y0, x1) - px(y0, x0));
        const float bot = px(y1, x0) + ax * (px(y1, x1) - px(y1, x0));
        out[c * static_cast<size_t>(dw) * dh + y * dw + x] = ((top + ay * (bot - top)) / 255.f - p.mean[c]) * p.inv_std[c];
      }
    }
  return out;
}

int check(int sw, int sh, int dw, int dh) {
  cv::Mat src(sh, sw, CV_8UC3);
  cv::randu(src, 0, 256);
  const size_t n = 3 * static_cast<size_t>(dw) * dh;
  unsigned char *d_in, *d_tmp;
  float *d_fused, *d_unfused;
  CUDA_CHECK(cudaMalloc(&d_in, src.total() * 3));
  CUDA_CHECK(cudaMalloc(&d_tmp, n));
  CUDA_CHECK(cudaMalloc(&d_fused, n * 4));
  CUDA_CHECK(cudaMalloc(&d_unfused, n * 4));
  CUDA_CHECK(cudaMemcpy(d_in, src.data, src.total() * 3, cudaMemcpyHostToDevice));
  preprocess_nchw(d_in, sw, sh, d_fused, dw, dh, kImageNet);
  preprocess_nchw_unfused(d_in, sw, sh, d_unfused, dw, dh, kImageNet, d_tmp);
  std::vector<float> fused(n), unfused(n);
  CUDA_CHECK(cudaMemcpy(fused.data(), d_fused, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(unfused.data(), d_unfused, n * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_fused); cudaFree(d_unfused);

  const std::vector<float> ref = cpu_reference(src, dw, dh, kImageNet);
  double vs_cpu = 0, vs_unfused = 0;
  for (size_t i = 0; i < n; ++i) {
    vs_cpu = std::max(vs_cpu, static_cast<double>(std::fabs(fused[i] - ref[i])));
    vs_unfused = std::max(vs_unfused, static_cast<double>(std::fabs(fused[i] - unfused[i])));
  }
  // Fused vs CPU: float rounding only. Fused vs unfused: the unfused path rounds to uint8
  // in between, i.e. up to 0.5/255 in pixel units, times 1/std (<= 1/0.224).
  const double unfused_bound = 0.5 / 255.0 / 0.224 + 1e-4;
  const bool ok = vs_cpu < 1e-3 && vs_unfused <= unfused_bound;
  std::printf("%-4s %5dx%-5d -> %4dx%-4d  fused vs CPU %.2e (< 1e-3), fused vs unfused %.2e (<= %.2e)\n",
              ok ? "ok" : "FAIL", sw, sh, dw, dh, vs_cpu, vs_unfused, unfused_bound);
  return !ok;
}

}  // namespace

int main() {
  int failures = 0;
  failures += check(1920, 1080, 224, 224) + check(1921, 1081, 640, 480) + check(64, 48, 128, 96) +
              check(33, 17, 7, 5) + check(1, 1, 3, 3);
  preprocess_nchw(nullptr, 0, 0, nullptr, 0, 0, kImageNet);  // degenerate sizes are no-ops
  std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
  return failures ? 1 : 0;
}
