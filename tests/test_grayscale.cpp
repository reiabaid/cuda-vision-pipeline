#include <opencv2/opencv.hpp>
#include "cvk.h"

int main() {
  // Odd size on purpose: exercises bounds handling.
  const int W = 1921, H = 1081;
  cv::Mat bgr(H, W, CV_8UC3);
  cv::randu(bgr, 0, 256);

  cv::Mat ref;
  cv::cvtColor(bgr, ref, cv::COLOR_BGR2GRAY);

  unsigned char *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, W * H * 3));
  CUDA_CHECK(cudaMalloc(&d_out, W * H));
  CUDA_CHECK(cudaMemcpy(d_in, bgr.data, W * H * 3, cudaMemcpyHostToDevice));
  grayscale_gpu(d_in, d_out, W, H);
  cv::Mat out(H, W, CV_8UC1);
  CUDA_CHECK(cudaMemcpy(out.data, d_out, W * H, cudaMemcpyDeviceToHost));

  cv::Mat diff;
  cv::absdiff(out, ref, diff);
  double maxdiff;
  cv::minMaxLoc(diff, nullptr, &maxdiff);
  std::printf("grayscale max abs diff vs OpenCV: %.0f\n", maxdiff);
  cudaFree(d_in); cudaFree(d_out);
  return maxdiff <= 1.0 ? 0 : 1;  // allow 1 LSB rounding difference
}
