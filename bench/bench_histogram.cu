// 256-bin histogram: ideal traffic is 1 B read per pixel (the 1 KB result is negligible).
// Uniform random input spreads atomics over all bins; constant input (all pixels in one bin)
// is the worst case for the global-atomics baseline and is deliberately not the headline.
#include "sweep.h"

int main(int argc, char** argv) {
  using Fn = void (*)(const unsigned char*, unsigned*, int, int, cudaStream_t);
  static Fn kShared = histogram256, kGlobal = histogram256_naive;
  auto wrap = [](Fn f) {
    return [f](const unsigned char* in, unsigned char* out, int w, int h, cudaStream_t s) {
      f(in, reinterpret_cast<unsigned*>(out), w, h, s);
    };
  };
  return bench::run_sweep(argc, argv,
                          {"histogram", 1, 1, 1.0, {{"global", wrap(kGlobal)}, {"shared", wrap(kShared)}}, /*random_input=*/true});
}
