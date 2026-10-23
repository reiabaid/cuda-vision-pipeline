#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>

#include "cvk.h"
#include "test_util.h"

using testutil::max_abs_diff;
using testutil::run_on_gpu;

namespace {

cv::Mat reference(const cv::Mat& src) {
  cv::Mat gx, gy, ax, ay, out;
  cv::Sobel(src, gx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REFLECT_101);
  cv::Sobel(src, gy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REFLECT_101);
  cv::convertScaleAbs(gx, ax);
  cv::convertScaleAbs(gy, ay);
  cv::add(ax, ay, out);  // saturating
  return out;
}

int check(int W, int H, const char* pattern) {
  cv::Mat src(H, W, CV_8UC1);
  if (std::string(pattern) == "random") cv::randu(src, 0, 256);
  else
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) src.at<uchar>(y, x) = ((x / 2 + y / 3) % 2) ? 255 : 0;
  const double d = max_abs_diff(
      run_on_gpu([](auto... a) { sobel_magnitude(a...); }, src), reference(src));
  std::printf("%-4s %-8s %5dx%-5d max diff vs OpenCV = %.0f\n", d == 0 ? "ok" : "FAIL", pattern, W, H, d);
  return d != 0;
}

}  // namespace

int main() {
  int failures = 0;
  for (auto [w, h] : {std::pair{1, 1}, {2, 2}, {3, 3}, {1, 20}, {20, 1}, {5, 5}, {33, 17}, {1921, 1081}})
    failures += check(w, h, "random");
  failures += check(33, 17, "checker") + check(1921, 1081, "checker");
  sobel_magnitude(nullptr, nullptr, 0, 5);  // degenerate sizes are no-ops
  std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
  return failures ? 1 : 0;
}
