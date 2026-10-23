#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

#include "cvk.h"
#include "test_util.h"

namespace {

using testutil::GpuFn;
using testutil::max_abs_diff;
using testutil::run_on_gpu;

struct Variant {
  const char* name;
  GpuFn fn;
};

const std::vector<Variant>& variants() {
  static const std::vector<Variant> v = {
      {"naive", [](auto... a) { gaussian_blur5_naive(a...); }},
      {"tiled", [](auto... a) { gaussian_blur5_tiled(a...); }},
  };
  return v;
}

int check(int W, int H, const char* pattern) {
  cv::Mat src(H, W, CV_8UC1);
  if (std::string(pattern) == "random") {
    cv::randu(src, 0, 256);
  } else if (std::string(pattern) == "checker") {  // maximum contrast, stresses rounding
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) src.at<uchar>(y, x) = ((x / 2 + y / 3) % 2) ? 255 : 0;
  } else {  // "white": saturated, result must stay 255
    src.setTo(255);
  }

  cv::Mat ref;
  cv::GaussianBlur(src, ref, cv::Size(5, 5), 1.0, 1.0, cv::BORDER_REFLECT_101);

  int failures = 0;
  for (const Variant& v : variants()) {
    const double d = max_abs_diff(run_on_gpu(v.fn, src), ref);
    if (d != 0.0) ++failures;
    std::printf("%-4s %-8s %-8s %5dx%-5d max diff vs OpenCV = %.0f\n", d == 0.0 ? "ok" : "FAIL",
                v.name, pattern, W, H, d);
  }
  return failures;
}

}  // namespace

int main() {
  int failures = 0;
  // Includes images narrower/shorter than the kernel (1, 2, 3 px) to exercise border
  // reflection, sizes that are not multiples of the block size, and large frames.
  for (auto [w, h] : {std::pair{1, 1}, {2, 2}, {3, 3}, {1, 20}, {20, 1}, {2, 9}, {5, 5}, {7, 3},
                      {33, 17}, {64, 64}, {1921, 1081}, {1920, 1080}}) {
    failures += check(w, h, "random");
  }
  failures += check(33, 17, "checker");
  failures += check(1921, 1081, "checker");
  failures += check(1920, 1080, "white");
  // Degenerate sizes must be a no-op, not a failed launch that kills the process.
  for (const Variant& v : variants()) { v.fn(nullptr, nullptr, 0, 5); v.fn(nullptr, nullptr, 5, -1); }
  std::printf("ok   degenerate dimensions are no-ops\n");
  std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
  return failures ? 1 : 0;
}
