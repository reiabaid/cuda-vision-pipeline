#pragma once
#include <opencv2/core.hpp>

#include <functional>

#include "cvk.h"

namespace testutil {

// Any 8-bit image kernel with the signature (d_in, d_out, width, height).
using GpuFn = std::function<void(const unsigned char*, unsigned char*, int, int)>;

// Uploads `in` (continuous 8-bit image), runs `fn`, downloads the result.
// `in_offset` shifts the device input pointer to test unaligned-input handling.
inline cv::Mat run_on_gpu(const GpuFn& fn, const cv::Mat& in, int out_channels = 1,
                          size_t in_offset = 0) {
  CV_Assert(in.isContinuous() && in.depth() == CV_8U);
  const size_t in_bytes = in.total() * in.elemSize();
  const size_t out_bytes = in.total() * out_channels;
  unsigned char *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, in_bytes + in_offset));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMemcpy(d_in + in_offset, in.data, in_bytes, cudaMemcpyHostToDevice));
  fn(d_in + in_offset, d_out, in.cols, in.rows);
  CUDA_CHECK(cudaDeviceSynchronize());
  cv::Mat out(in.rows, in.cols, CV_8UC(out_channels));
  CUDA_CHECK(cudaMemcpy(out.data, d_out, out_bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
  return out;
}

inline double max_abs_diff(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  double m;
  cv::minMaxLoc(diff.reshape(1), nullptr, &m);
  return m;
}

}  // namespace testutil
