#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cvk.h"
#include "test_util.h"

namespace {

using testutil::GpuFn;
using testutil::max_abs_diff;
using testutil::run_on_gpu;
using GrayFn = GpuFn;

int check(int W, int H, size_t in_offset = 0) {
  cv::Mat bgr(H, W, CV_8UC3);
  cv::randu(bgr, 0, 256);
  cv::Mat ref;
  cv::cvtColor(bgr, ref, cv::COLOR_BGR2GRAY);

  const GrayFn naive = [](auto... a) { grayscale_gpu(a...); };
  const GrayFn vec4 = [](auto... a) { grayscale_gpu_vec4(a...); };
  const cv::Mat out_naive = run_on_gpu(naive, bgr, 1, in_offset);
  const cv::Mat out_vec4 = run_on_gpu(vec4, bgr, 1, in_offset);

  // A misaligned input shifts which bytes the kernels see, so only the two kernels
  // can be compared with each other there, not with OpenCV.
  const double vs_cv_naive = in_offset ? 0.0 : max_abs_diff(out_naive, ref);
  const double vs_cv_vec4 = in_offset ? 0.0 : max_abs_diff(out_vec4, ref);
  const double naive_vs_vec4 = max_abs_diff(out_naive, out_vec4);

  // Exact match required: the kernels use OpenCV's own fixed-point coefficients
  // (3735/19235/9798, 15-bit), verified against OpenCV 4.14.0. Other OpenCV versions
  // may round differently; if this fails there, re-derive the coefficients first.
  const bool ok = vs_cv_naive == 0.0 && vs_cv_vec4 == 0.0 && naive_vs_vec4 == 0.0;
  std::printf("%-4s %5dx%-5d offset=%zu  naive-vs-cv=%.0f  vec4-vs-cv=%.0f  naive-vs-vec4=%.0f\n",
              ok ? "ok" : "FAIL", W, H, in_offset, vs_cv_naive, vs_cv_vec4, naive_vs_vec4);
  return ok ? 0 : 1;
}

// Every one of the 2^24 possible BGR values appears exactly once.
int check_all_colors() {
  const int N = 4096;
  cv::Mat bgr(N, N, CV_8UC3);
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      const unsigned v = static_cast<unsigned>(y) * N + x;
      bgr.at<cv::Vec3b>(y, x) = cv::Vec3b(v & 255, (v >> 8) & 255, (v >> 16) & 255);
    }
  cv::Mat ref;
  cv::cvtColor(bgr, ref, cv::COLOR_BGR2GRAY);

  const GrayFn naive = [](auto... a) { grayscale_gpu(a...); };
  const GrayFn vec4 = [](auto... a) { grayscale_gpu_vec4(a...); };
  const double d_naive = max_abs_diff(run_on_gpu(naive, bgr), ref);
  const double d_vec4 = max_abs_diff(run_on_gpu(vec4, bgr), ref);
  const bool ok = d_naive == 0.0 && d_vec4 == 0.0;
  std::printf("%-4s all 2^24 colors  naive-vs-cv=%.0f  vec4-vs-cv=%.0f\n", ok ? "ok" : "FAIL",
              d_naive, d_vec4);
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int failures = 0;
  // Sizes chosen so n mod 4 covers 0..3, plus n < 4 (tail only) and large odd frames.
  for (auto [w, h] : {std::pair{1, 1}, {3, 1}, {5, 1}, {7, 3}, {4, 4}, {33, 17}, {1921, 1081},
                      {1920, 1080}, {3840, 2160}}) {
    failures += check(w, h);
  }
  failures += check(1921, 1081, /*in_offset=*/1);  // exercises the unaligned fallback
  failures += check_all_colors();
  grayscale_gpu(nullptr, nullptr, 0, 5);       // degenerate sizes must be no-ops
  grayscale_gpu_vec4(nullptr, nullptr, 5, -1);
  std::printf("ok   degenerate dimensions are no-ops\n");
  std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
  return failures ? 1 : 0;
}
