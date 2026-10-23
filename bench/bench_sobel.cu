// Sobel magnitude, single channel: ideal traffic is 1 B read + 1 B written per pixel.
#include "sweep.h"

int main(int argc, char** argv) {
  return bench::run_sweep(argc, argv, {"sobel", 1, 1, 2.0, {{"tiled", sobel_magnitude}}});
}
