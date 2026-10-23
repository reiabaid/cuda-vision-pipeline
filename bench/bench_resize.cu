// Bilinear 2x downscale of a BGR image. Ideal traffic per *source* pixel: 3 B read
// (a 2x downscale touches every source pixel once) + 0.75 B written.
#include "sweep.h"

int main(int argc, char** argv) {
  auto resize2x = [](const unsigned char* in, unsigned char* out, int w, int h, cudaStream_t s) {
    resize_bilinear_bgr(in, w, h, out, w / 2, h / 2, s);
  };
  return bench::run_sweep(argc, argv, {"resize", 3, 3, 3.75, {{"bilinear", resize2x}}});
}
