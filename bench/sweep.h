#pragma once
// Shared driver for per-kernel benchmarks: sweeps image sizes, times every variant,
// prints a table and writes a CSV.
// Usage of any benchmark built on this: bench_x [--warm] [--csv <path>]
//   default: cold L2 before every iteration (DRAM-bound numbers)
//   --warm : no flush (shows L2 effects on small images)
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "bench_util.h"

namespace bench {

struct Size {
  int w, h;
};

using KernelFn = std::function<void(const unsigned char*, unsigned char*, int, int, cudaStream_t)>;

struct Variant {
  const char* name;
  KernelFn fn;
};

struct Sweep {
  std::string kernel;           // names the default CSV: results/<kernel>.csv
  int in_bytes_per_pixel;       // size of the input buffer
  int out_bytes_per_pixel;      // size of the output buffer
  double traffic_bytes_per_pixel;  // ideal DRAM traffic, used for GB/s and % of peak
  std::vector<Variant> variants;
  bool random_input = false;      // pseudo-random bytes instead of a constant (data-dependent kernels)
};

inline const std::vector<Size>& default_sizes() {
  static const std::vector<Size> s = {{256, 256},   {512, 512},   {1024, 1024},
                                      {1920, 1080}, {3840, 2160}, {7680, 4320}};
  return s;
}

// Cheap integer-hash fill; content only matters for data-dependent kernels (histogram).
__global__ void fill_pseudo_random(unsigned char* p, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned x = static_cast<unsigned>(i) * 2654435761u;
  x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
  p[i] = static_cast<unsigned char>(x);
}

inline int run_sweep(int argc, char** argv, const Sweep& sweep) {
  bool warm = false;
  std::string csv_path = "results/" + sweep.kernel + ".csv";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--warm")) warm = true;
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else { std::fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
  }

  const DeviceInfo dev = query_device();
  std::printf("%s | GPU: %s | peak DRAM: %.1f GB/s | L2: %.1f MB | cache: %s\n\n",
              sweep.kernel.c_str(), dev.name.c_str(), dev.peak_gbps, dev.l2_bytes / 1048576.0,
              warm ? "warm" : "cold");

  L2Flusher flusher(dev.l2_bytes);
  Config cfg;
  if (!warm) cfg.flusher = &flusher;

  std::filesystem::path out(csv_path);
  if (out.has_parent_path()) std::filesystem::create_directories(out.parent_path());
  std::ofstream csv(out);
  csv << "device,cache,kernel,width,height,bytes,median_ms,min_ms,p95_ms,gbps,pct_peak,speedup\n";

  std::printf("%11s %9s %10s %10s %10s %10s %10s %8s\n", "size", "variant", "median ms",
              "min ms", "p95 ms", "GB/s", "% peak", "speedup");

  for (const Size& s : default_sizes()) {
    const size_t pixels = static_cast<size_t>(s.w) * s.h;
    unsigned char *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, pixels * sweep.in_bytes_per_pixel));
    CUDA_CHECK(cudaMalloc(&d_out, pixels * sweep.out_bytes_per_pixel));
    // Constant data by default: most kernels here are data-independent.
    const size_t in_bytes = pixels * sweep.in_bytes_per_pixel;
    if (sweep.random_input) fill_pseudo_random<<<static_cast<unsigned>((in_bytes + 255) / 256), 256>>>(d_in, in_bytes);
    else CUDA_CHECK(cudaMemset(d_in, 0x7f, in_bytes));

    const double bytes = static_cast<double>(pixels) * sweep.traffic_bytes_per_pixel;
    double baseline_ms = 0.0;

    for (const Variant& v : sweep.variants) {
      const Stats st = time_ms([&] { v.fn(d_in, d_out, s.w, s.h, cfg.stream); }, cfg);
      if (baseline_ms == 0.0) baseline_ms = st.median_ms;

      const double gbps = bytes / (st.median_ms * 1e-3) / 1e9;
      const double pct = 100.0 * gbps / dev.peak_gbps;
      const double speedup = baseline_ms / st.median_ms;

      char label[32];
      std::snprintf(label, sizeof label, "%dx%d", s.w, s.h);
      std::printf("%11s %9s %10.4f %10.4f %10.4f %10.1f %9.1f%% %7.2fx\n", label, v.name,
                  st.median_ms, st.min_ms, st.p95_ms, gbps, pct, speedup);
      csv << '"' << dev.name << "\"," << (warm ? "warm" : "cold") << ',' << sweep.kernel << '/'
          << v.name << ',' << s.w << ',' << s.h << ',' << static_cast<long long>(bytes) << ','
          << st.median_ms << ',' << st.min_ms << ',' << st.p95_ms << ',' << gbps << ',' << pct
          << ',' << speedup << '\n';
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
  }

  std::printf("\nCSV written to %s\n", csv_path.c_str());
  return 0;
}

}  // namespace bench
