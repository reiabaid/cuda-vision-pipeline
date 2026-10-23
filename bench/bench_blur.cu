// 5x5 Gaussian blur, single channel: ideal traffic is 1 B read + 1 B written per pixel.
// "% peak" is therefore measured against that ideal, so a kernel that re-reads input
// from DRAM (or spills an intermediate image) cannot reach 100%.
#include "sweep.h"

int main(int argc, char** argv) {
  return bench::run_sweep(argc, argv, {"blur", 1, 1, 2.0,
                           {{"naive", gaussian_blur5_naive}, {"tiled", gaussian_blur5_tiled},
                            {"tiled4", gaussian_blur5_tiled4}}});
}
