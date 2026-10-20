// Grayscale: reads 3 B and writes 1 B per pixel.
#include "sweep.h"

int main(int argc, char** argv) {
  return bench::run_sweep(argc, argv,
                          {"grayscale", 3, 1, 4.0,
                           {{"naive", grayscale_gpu}, {"vec4", grayscale_gpu_vec4}}});
}
