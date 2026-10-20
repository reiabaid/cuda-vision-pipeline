#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "cvk.h"

namespace bench {

struct Stats {
  float median_ms;
  float min_ms;
  float p95_ms;
};

struct DeviceInfo {
  std::string name;
  double peak_gbps;   // theoretical DRAM bandwidth
  size_t l2_bytes;
};

// Peak = memory clock * 2 (double data rate) * bus width in bytes.
inline DeviceInfo query_device(int dev = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int clock_khz = 0, bus_bits = 0, l2 = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  double peak = 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  return {prop.name, peak, static_cast<size_t>(l2)};
}

// Reads one buffer to evict whatever is in L2. The sum is written only if it hits
// an impossible value, which keeps the compiler from deleting the loads.
__global__ void read_flush_kernel(const uint4* __restrict__ buf, size_t n, unsigned* sink) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  unsigned acc = 0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint4 v = buf[i];
    acc += v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) *sink = acc;
}

// Reads twice the L2 size so the next kernel starts with a cold cache and its
// reads must come from DRAM, not from data left by the previous iteration.
// Reading (not writing) matters: a write-based flush leaves dirty lines that must be
// written back to DRAM while the timed kernel runs, inflating its time.
class L2Flusher {
 public:
  explicit L2Flusher(size_t l2_bytes) : bytes_(2 * l2_bytes) {
    CUDA_CHECK(cudaMalloc(&buf_, bytes_));
    CUDA_CHECK(cudaMalloc(&sink_, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(buf_, 0xA5, bytes_));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  ~L2Flusher() {
    cudaFree(buf_);
    cudaFree(sink_);
  }
  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;

  void flush(cudaStream_t stream) {
    read_flush_kernel<<<256, 256, 0, stream>>>(static_cast<const uint4*>(buf_),
                                               bytes_ / sizeof(uint4), sink_);
    CUDA_CHECK(cudaGetLastError());
  }

 private:
  void* buf_ = nullptr;
  unsigned* sink_ = nullptr;
  size_t bytes_;
};

struct Config {
  int warmup = 10;
  int iters = 200;
  cudaStream_t stream = 0;
  L2Flusher* flusher = nullptr;  // null = warm-cache timing
};

// Times `launch()` with GPU-side events. Each iteration gets its own event pair,
// so one slow iteration (clock dip, driver hiccup) shows up in p95, not the median.
template <class Launch>
Stats time_ms(Launch&& launch, const Config& cfg = {}) {
  for (int i = 0; i < cfg.warmup; ++i) launch();
  CUDA_CHECK(cudaStreamSynchronize(cfg.stream));

  std::vector<cudaEvent_t> start(cfg.iters), stop(cfg.iters);
  for (int i = 0; i < cfg.iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&stop[i]));
  }

  for (int i = 0; i < cfg.iters; ++i) {
    if (cfg.flusher) cfg.flusher->flush(cfg.stream);
    CUDA_CHECK(cudaEventRecord(start[i], cfg.stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop[i], cfg.stream));
  }
  CUDA_CHECK(cudaEventSynchronize(stop[cfg.iters - 1]));

  std::vector<float> ms(cfg.iters);
  for (int i = 0; i < cfg.iters; ++i) {
    CUDA_CHECK(cudaEventElapsedTime(&ms[i], start[i], stop[i]));
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(stop[i]));
  }

  std::sort(ms.begin(), ms.end());
  const int n = cfg.iters;
  float median = (n % 2) ? ms[n / 2] : 0.5f * (ms[n / 2 - 1] + ms[n / 2]);
  float p95 = ms[std::min(n - 1, static_cast<int>(0.95 * n))];
  return {median, ms.front(), p95};
}

}  // namespace bench
