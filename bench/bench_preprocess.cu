// ML preprocessing (2x downscale, BGR u8 HWC -> normalized RGB float32 CHW): fused kernel
// vs the same work as two kernels with a uint8 intermediate. Ideal traffic per source pixel
// is the fused minimum: 3 B read + 3 B written (0.75 px/px x 3 channels x 4 B). The unfused
// path moves an extra 0.75 B written + 0.75 B read, so it cannot reach 100% of this ideal.
#include "sweep.h"

int main(int argc, char** argv) {
  const NormParams params = {{0.485f, 0.456f, 0.406f}, {1 / 0.229f, 1 / 0.224f, 1 / 0.225f}};

  unsigned char* tmp = nullptr;  // intermediate image for the unfused path (largest sweep size)
  CUDA_CHECK(cudaMalloc(&tmp, 3ull * 3840 * 2160));

  auto fused = [&](const unsigned char* in, unsigned char* out, int w, int h, cudaStream_t s) {
    preprocess_nchw(in, w, h, reinterpret_cast<float*>(out), w / 2, h / 2, params, s);
  };
  auto unfused = [&](const unsigned char* in, unsigned char* out, int w, int h, cudaStream_t s) {
    preprocess_nchw_unfused(in, w, h, reinterpret_cast<float*>(out), w / 2, h / 2, params, tmp, s);
  };
  const int rc = bench::run_sweep(argc, argv, {"preprocess", 3, 3, 6.0, {{"unfused", unfused}, {"fused", fused}}});
  cudaFree(tmp);
  return rc;
}
