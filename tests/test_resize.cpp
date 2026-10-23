#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cvk.h"
#include "test_util.h"

namespace {

// Resize changes the image size, so this does not fit testutil::run_on_gpu.
cv::Mat gpu_resize(const cv::Mat& src, int dw, int dh) {
  unsigned char *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, src.total() * 3));
  CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(dw) * dh * 3));
  CUDA_CHECK(cudaMemcpy(d_in, src.data, src.total() * 3, cudaMemcpyHostToDevice));
  resize_bilinear_bgr(d_in, src.cols, src.rows, d_out, dw, dh);
  CUDA_CHECK(cudaDeviceSynchronize());
  cv::Mat out(dh, dw, CV_8UC3);
  CUDA_CHECK(cudaMemcpy(out.data, d_out, out.total() * 3, cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
  return out;
}

// OpenCV uses 11-bit fixed-point weights; this kernel uses float, so up to 1 LSB of
// difference is expected. We require max diff <= 1 and report how often it differs.
int check(int sw, int sh, int dw, int dh) {
  cv::Mat src(sh, sw, CV_8UC3);
  cv::randu(src, 0, 256);
  cv::Mat ref;
  cv::resize(src, ref, cv::Size(dw, dh), 0, 0, cv::INTER_LINEAR);
  const cv::Mat out = gpu_resize(src, dw, dh);
  const double d = testutil::max_abs_diff(out, ref);
  cv::Mat ne;
  cv::compare(out, ref, ne, cv::CMP_NE);
  const double frac = 100.0 * cv::countNonZero(ne.reshape(1)) / (ref.total() * 3);
  std::printf("%-4s %5dx%-5d -> %4dx%-4d max diff %.0f, %.2f%% of values differ\n",
              d <= 1 ? "ok" : "FAIL", sw, sh, dw, dh, d, frac);
  return d > 1;
}

}  // namespace

int main() {
  int failures = 0;
  failures += check(1920, 1080, 960, 540);   // 2x down
  failures += check(1920, 1080, 224, 224);   // typical ML input, non-uniform scale
  failures += check(1921, 1081, 640, 480);   // odd source size
  failures += check(64, 48, 128, 96);        // 2x up
  failures += check(33, 17, 100, 60);        // up, non-integer ratio
  failures += check(1, 1, 4, 4) + check(5, 1, 3, 1) + check(4, 4, 1, 1);  // degenerate shapes
  resize_bilinear_bgr(nullptr, 0, 0, nullptr, 0, 0);  // degenerate sizes are no-ops
  std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
  return failures ? 1 : 0;
}
