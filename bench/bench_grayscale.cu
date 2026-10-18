// Times the grayscale kernel across image sizes and reports achieved bandwidth.
// Usage: bench_grayscale [--warm] [--csv <path>]
//   default: cold L2 before every iteration (DRAM-bound numbers)
//   --warm : no flush (shows L2 effects on small images)
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "bench_util.h"

struct Size {
  int w, h;
};

int main(int argc, char** argv) {
  bool warm = false;
  std::string csv_path = "results/grayscale.csv";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--warm")) warm = true;
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else { std::fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
  }

  const bench::DeviceInfo dev = bench::query_device();
  std::printf("GPU: %s | peak DRAM bandwidth: %.1f GB/s | L2: %.1f MB | cache: %s\n\n",
              dev.name.c_str(), dev.peak_gbps, dev.l2_bytes / 1048576.0,
              warm ? "warm" : "cold");

  const std::vector<Size> sizes = {{256, 256},   {512, 512},   {1024, 1024},
                                   {1920, 1080}, {3840, 2160}, {7680, 4320}};

  bench::L2Flusher flusher(dev.l2_bytes);
  bench::Config cfg;
  if (!warm) cfg.flusher = &flusher;

  std::filesystem::path out(csv_path);
  if (out.has_parent_path()) std::filesystem::create_directories(out.parent_path());
  std::ofstream csv(out);
  csv << "device,cache,width,height,bytes,median_ms,min_ms,p95_ms,gbps,pct_peak\n";

  std::printf("%11s %10s %10s %10s %10s %10s\n", "size", "median ms", "min ms", "p95 ms",
              "GB/s", "% peak");

  for (const Size& s : sizes) {
    const size_t pixels = static_cast<size_t>(s.w) * s.h;
    unsigned char *d_bgr = nullptr, *d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bgr, pixels * 3));
    CUDA_CHECK(cudaMalloc(&d_gray, pixels));
    // Pixel values do not affect this kernel's speed; just avoid reading uninitialized memory.
    CUDA_CHECK(cudaMemset(d_bgr, 0x7f, pixels * 3));

    const bench::Stats st =
        bench::time_ms([&] { grayscale_gpu(d_bgr, d_gray, s.w, s.h, cfg.stream); }, cfg);

    // 3 B read + 1 B written per pixel.
    const double bytes = static_cast<double>(pixels) * 4.0;
    const double gbps = bytes / (st.median_ms * 1e-3) / 1e9;
    const double pct = 100.0 * gbps / dev.peak_gbps;

    char label[32];
    std::snprintf(label, sizeof label, "%dx%d", s.w, s.h);
    std::printf("%11s %10.4f %10.4f %10.4f %10.1f %9.1f%%\n", label, st.median_ms, st.min_ms,
                st.p95_ms, gbps, pct);
    csv << '"' << dev.name << "\"," << (warm ? "warm" : "cold") << ',' << s.w << ',' << s.h
        << ',' << static_cast<long long>(bytes) << ',' << st.median_ms << ',' << st.min_ms << ','
        << st.p95_ms << ',' << gbps << ',' << pct << '\n';

    CUDA_CHECK(cudaFree(d_bgr));
    CUDA_CHECK(cudaFree(d_gray));
  }

  std::printf("\nCSV written to %s\n", csv_path.c_str());
  return 0;
}
