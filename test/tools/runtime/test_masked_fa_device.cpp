// test/tools/runtime/test_masked_fa_device.cpp
//
// DEVICE-mode unit test for causal-masked FlashAttention.
//
// In HOST mode run_FlashAttentionScore calls sdpa_cpu, which APPLIES the
// additive causal mask correctly (this is the sim source-of-truth used by the
// GPT-2 / encoder / BERT e2e). On a real device the SAME call dispatches to
// aclnnFlashAttentionScore. The device branch historically hardcoded
// attenMask=nullptr + nextTokens=65536 (full bidirectional attention), so a
// causal mask was silently ignored — correct on sim, WRONG on the NPU.
//
// This test pins that: it runs the host reference (masked) and the device path
// on the SAME causal inputs and asserts they agree. It FAILS before the device
// causal-mask fix and PASSES after.
//
// Build + run ON the dev host (real 910C, CANN present):
//   TK=/data/nyh/Ascend/cann-9.1.0; source $TK/set_env.sh
//   g++ -std=c++17 -I include -I $TK/include \
//       test/tools/runtime/test_masked_fa_device.cpp lib/Runtime/AclnnOps.cpp \
//       -L $TK/lib64 -Wl,-rpath,$TK/lib64 \
//       -lascendcl -lnnopbase -lopapi -lopapi_nn -lopapi_math \
//       -lopapi_transformer -Wl,--allow-shlib-undefined -o /tmp/t_mfa
//   ASCEND_DEVICE_ID=7 /tmp/t_mfa

#include "Runtime/AclnnOps.h"
#include "acl/acl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace mlir::runtime::aclnn;

// Build an f32 host-memory TensorInfo over caller-owned data.
static TensorInfo mk(std::vector<int64_t> shape, float *data) {
  TensorInfo t;
  t.rank = static_cast<int>(shape.size());
  for (int i = 0; i < t.rank; ++i) t.shape[i] = shape[i];
  rowMajorStrides(t.shape, t.rank, t.strides);
  t.dtype = 0;  // ACL_FLOAT (f32)
  t.data = data;
  return t;
}

#define LOG(...) do { std::fprintf(stderr, __VA_ARGS__); std::fflush(stderr); } while (0)

int main() {
  // GPT-2-representative, FA-hardware-friendly shapes: 12 heads, S=64, D=64.
  const int Hh = 12, S = 64, D = 64;
  const int NEL = Hh * S * D;

  std::vector<float> q(NEL), k(NEL), v(NEL);
  for (int i = 0; i < NEL; ++i) {
    q[i] = 0.01f * ((i * 7) % 13);
    k[i] = 0.01f * ((i * 5) % 11);
    v[i] = 0.10f * ((i * 3) % 9);
  }
  // Additive causal mask [1,1,S,S]: 0 on/below diagonal, -1e9 above.
  std::vector<float> mask(S * S, 0.0f);
  for (int i = 0; i < S; ++i)
    for (int j = 0; j < S; ++j)
      if (j > i) mask[i * S + j] = -1e9f;

  TensorInfo qt = mk({1, Hh, S, D}, q.data());
  TensorInfo kt = mk({1, Hh, S, D}, k.data());
  TensorInfo vt = mk({1, Hh, S, D}, v.data());
  TensorInfo mt = mk({1, 1, S, S}, mask.data());

  // 1) HOST reference (sdpa_cpu applies the causal mask).
  LOG("[mfa] host sdpa_cpu ...\n");
  setHostMode(true);
  TensorInfo hostOut;
  run_FlashAttentionScore(qt, kt, vt, mt, qt, &hostOut, nullptr);
  LOG("[mfa] host done\n");

  // 2) DEVICE path.
  setHostMode(false);
  LOG("[mfa] aclInit ...\n");
  if (aclInit(nullptr) != ACL_SUCCESS) {
    std::printf("aclInit failed — cannot run device test\n");
    return 2;
  }
  int devId = 0;
  if (const char *e = std::getenv("ASCEND_DEVICE_ID")) devId = std::atoi(e);
  LOG("[mfa] aclrtSetDevice(%d) ...\n", devId);
  aclrtSetDevice(devId);
  aclrtStream stream = nullptr;
  aclrtCreateStream(&stream);
  LOG("[mfa] device run_FlashAttentionScore ...\n");
  TensorInfo devOut;
  run_FlashAttentionScore(qt, kt, vt, mt, qt, &devOut, stream);
  LOG("[mfa] sync ...\n");
  aclrtSynchronizeStream(stream);
  LOG("[mfa] device done\n");

  // 3) Compare device vs host (causal) output.
  const float *h = (const float *)hostOut.data;
  const float *d = (const float *)devOut.data;
  float maxDiff = 0.0f;
  for (int i = 0; i < S * D; ++i)
    maxDiff = std::fmax(maxDiff, std::fabs(h[i] - d[i]));
  std::printf("masked-FA causal: max_abs_diff(host,device) = %.6g\n", maxDiff);

  // Show row 0 (the row a missing causal mask corrupts most: bidirectional
  // would attend to all S positions instead of only position 0).
  std::printf("  host row0[0..3]: %.4f %.4f %.4f %.4f\n", h[0], h[1], h[2], h[3]);
  std::printf("  dev  row0[0..3]: %.4f %.4f %.4f %.4f\n", d[0], d[1], d[2], d[3]);

  freeTensor(&hostOut);
  freeTensor(&devOut);
  aclrtDestroyStream(stream);
  aclrtResetDevice(devId);
  aclFinalize();

  bool pass = maxDiff < 1e-3f;
  std::printf("%s\n", pass ? "MASKED-FA DEVICE PASS" : "MASKED-FA DEVICE FAIL");
  return pass ? 0 : 1;
}
