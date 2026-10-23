#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

#include "cvk.h"

namespace {

std::vector<unsigned> run(void (*fn)(const unsigned char*, unsigned*, int, int, cudaStream_t),
                          const cv::Mat& img, size_t offset) {
  const size_t n = img.total();
  unsigned char* d_in;
  unsigned* d_hist;
  CUDA_CHECK(cudaMalloc(&d_in, n + offset));
  CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(d_hist, 0xff, 256 * sizeof(unsigned)));  // must be zeroed by the call
  CUDA_CHECK(cudaMemcpy(d_in + offset, img.data, n, cudaMemcpyHostToDevice));
  fn(d_in + offset, d_hist, img.cols, img.rows, 0);
  std::vector<unsigned> h(256);
  CUDA_CHECK(cudaMemcpy(h.data(), d_hist, 256 * sizeof(unsigned), cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_hist);
  return h;
}

int check(int W, int H, const char* kind, size_t offset = 0) {
  cv::Mat img(H, W, CV_8UC1);
  if (kind[0] == 'r') cv::randu(img, 0, 256);
  else img.setTo(kind[0] == 'z' ? 0 : 255);  // all-same pixels: worst-case atomic contention
  cv::Mat ref;
  const int bins = 256;
  const float range[] = {0, 256};
  const float* ranges[] = {range};
  int ch = 0;
  cv::calcHist(&img, 1, &ch, cv::Mat(), ref, 1, &bins, ranges);

  int failures = 0;
  for (auto [name, fn] : {std::pair<const char*, decltype(&histogram256)>{"shared", histogram256},
                          {"global", histogram256_naive}}) {
    const std::vector<unsigned> h = run(fn, img, offset);
    bool ok = true;
    for (int i = 0; i < 256; ++i) ok &= h[i] == static_cast<unsigned>(ref.at<float>(i));
    failures += !ok;
    std::printf("%-4s %-6s %-6s %5dx%-5d offset=%zu\n", ok ? "ok" : "FAIL", name, kind, W, H, offset);
  }
  return failures;
}

}  // namespace

int main() {
  int failures = 0;
  for (auto [w, h] : {std::pair{1, 1}, {3, 1}, {5, 3}, {33, 17}, {1921, 1081}, {3840, 2160}})
    failures += check(w, h, "random");
  failures += check(1921, 1081, "zeros") + check(1921, 1081, "full");
  failures += check(1921, 1081, "random", /*offset=*/1);  // unaligned input -> fallback path
  histogram256(nullptr, nullptr, 0, 0);                     // degenerate sizes are no-ops
  std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
  return failures ? 1 : 0;
}
