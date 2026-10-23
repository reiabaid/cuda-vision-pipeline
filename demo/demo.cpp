// End-to-end demo: BGR image -> grayscale -> Gaussian blur -> Sobel edges on the GPU, each
// stage verified against OpenCV on the CPU, with timings.
// Usage: cvk_demo [image] [--out <dir>]     (no image: a synthetic 1920x1080 test scene)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "cvk.h"

namespace {

cv::Mat synthetic_scene(int w, int h) {
  cv::Mat img(h, w, CV_8UC3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) img.at<cv::Vec3b>(y, x) = {uchar(255 * x / w), uchar(255 * y / h), 128};
  cv::circle(img, {w / 3, h / 2}, h / 4, {255, 255, 255}, -1);
  cv::rectangle(img, {w / 2, h / 4}, {w * 4 / 5, h * 3 / 4}, {20, 20, 20}, 12);
  cv::putText(img, "CUDA", {w / 8, h / 6}, cv::FONT_HERSHEY_SIMPLEX, 4, {0, 0, 255}, 8);
  cv::Mat noise(h, w, CV_8UC3);
  cv::randn(noise, 0, 6);
  return img + noise;
}

cv::Mat cpu_sobel(const cv::Mat& gray) {
  cv::Mat gx, gy, ax, ay, out;
  cv::Sobel(gray, gx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REFLECT_101);
  cv::Sobel(gray, gy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REFLECT_101);
  cv::convertScaleAbs(gx, ax);
  cv::convertScaleAbs(gy, ay);
  cv::add(ax, ay, out);
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <class F>
double median_ms(F&& f, int runs) {
  std::vector<double> t;
  for (int i = 0; i < runs; ++i) {
    const auto a = std::chrono::steady_clock::now();
    f();
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
  }
  return median(t);
}

}  // namespace

int main(int argc, char** argv) {
  std::string path, out_dir;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--out" && i + 1 < argc) out_dir = argv[++i];
    else path = argv[i];
  }
  cv::Mat bgr = path.empty() ? synthetic_scene(1920, 1080) : cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) { std::fprintf(stderr, "cannot read image: %s\n", path.c_str()); return 1; }
  if (!bgr.isContinuous()) bgr = bgr.clone();
  const int W = bgr.cols, H = bgr.rows;
  const size_t px = static_cast<size_t>(W) * H;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s | image: %dx%d %s\n\n", prop.name, W, H, path.empty() ? "(synthetic)" : path.c_str());

  unsigned char *d_bgr, *d_gray, *d_blur, *d_edge;
  CUDA_CHECK(cudaMalloc(&d_bgr, px * 3));
  CUDA_CHECK(cudaMalloc(&d_gray, px));
  CUDA_CHECK(cudaMalloc(&d_blur, px));
  CUDA_CHECK(cudaMalloc(&d_edge, px));
  auto gpu_pipeline = [&] {
    grayscale_gpu_vec4(d_bgr, d_gray, W, H);
    gaussian_blur5_tiled(d_gray, d_blur, W, H);
    sobel_magnitude(d_blur, d_edge, W, H);
  };
  CUDA_CHECK(cudaMemcpy(d_bgr, bgr.data, px * 3, cudaMemcpyHostToDevice));
  gpu_pipeline();
  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness: every stage against OpenCV, each fed the same input as the GPU stage.
  cv::Mat gray, blur, edge, g_gray(H, W, CV_8UC1), g_blur(H, W, CV_8UC1), g_edge(H, W, CV_8UC1);
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, {5, 5}, 1.0, 1.0, cv::BORDER_REFLECT_101);
  edge = cpu_sobel(blur);
  CUDA_CHECK(cudaMemcpy(g_gray.data, d_gray, px, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g_blur.data, d_blur, px, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g_edge.data, d_edge, px, cudaMemcpyDeviceToHost));
  auto diff = [](const cv::Mat& a, const cv::Mat& b) { return cv::norm(a, b, cv::NORM_INF); };
  const double d1 = diff(g_gray, gray), d2 = diff(g_blur, blur), d3 = diff(g_edge, edge);
  std::printf("Verification vs OpenCV (max abs diff, 0 = bit-exact): grayscale %.0f | blur %.0f | sobel %.0f  -> %s\n\n",
              d1, d2, d3, (d1 + d2 + d3 == 0) ? "PASS" : "FAIL");

  // Timing: GPU kernels only (CUDA events), GPU end-to-end (upload + kernels + download), CPU OpenCV.
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<double> kernel_ms;
  for (int i = 0; i < 110; ++i) {
    CUDA_CHECK(cudaEventRecord(e0));
    gpu_pipeline();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (i >= 10) kernel_ms.push_back(ms);  // first 10 are warm-up
  }
  const double gpu_kernels = median(kernel_ms);
  const double gpu_e2e = median_ms([&] {
    CUDA_CHECK(cudaMemcpy(d_bgr, bgr.data, px * 3, cudaMemcpyHostToDevice));
    gpu_pipeline();
    CUDA_CHECK(cudaMemcpy(g_edge.data, d_edge, px, cudaMemcpyDeviceToHost));
  }, 30);
  const double cpu = median_ms([&] {
    cv::Mat g, b;
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(g, b, {5, 5}, 1.0, 1.0, cv::BORDER_REFLECT_101);
    cpu_sobel(b);
  }, 30);

  std::printf("%-44s %9s\n", "Pipeline: grayscale -> blur 5x5 -> Sobel", "median ms");
  std::printf("%-44s %9.3f\n", "GPU, kernels only", gpu_kernels);
  std::printf("%-44s %9.3f   (PCIe copies dominate)\n", "GPU, end-to-end (upload + kernels + download)", gpu_e2e);
  std::printf("%-44s %9.3f   (%d OpenCV thread(s))\n", "CPU, OpenCV", cpu, cv::getNumThreads());
  std::printf("\nKernel-only speedup vs CPU: %.1fx | end-to-end: %.1fx\n", cpu / gpu_kernels, cpu / gpu_e2e);

  if (!out_dir.empty()) {
    std::filesystem::create_directories(out_dir);
    cv::imwrite(out_dir + "/1_gray.png", g_gray);
    cv::imwrite(out_dir + "/2_blur.png", g_blur);
    cv::imwrite(out_dir + "/3_edges.png", g_edge);
    std::printf("Wrote stage images to %s\n", out_dir.c_str());
  }
  cudaFree(d_bgr); cudaFree(d_gray); cudaFree(d_blur); cudaFree(d_edge);
  return (d1 + d2 + d3 == 0) ? 0 : 1;
}
