// test/tools/runtime/test_aclnn_ops.cpp
//
// Unit tests for the aclnn run_* wrappers (run_Matmul / run_LayerNorm /
// run_Transpose / run_FlashAttentionScore).
//
// In HOST mode (setHostMode(true)) the wrappers run a CPU reference. That
// reference is the source of truth the network_runner uses on the CANN
// simulator (aclInit fails → host mode), so keeping it correct is what makes
// the encoder / BERT e2e match PyTorch. This file pins that behaviour and is
// runnable WITHOUT an NPU.
//
// On a real Ascend device the SAME ops dispatch to aclnn kernels instead; that
// path is exercised by the --device runs (see DEVICE NOTE at the bottom). The
// expected values here were hand-computed and are dtype/backend agnostic, so
// the device path — once implemented — is validated against the same numbers.
//
// Build (no NPU needed — host CPU reference):
//   c++ -std=c++17 -I include \
//       test/tools/runtime/test_aclnn_ops.cpp lib/Runtime/AclnnOps.cpp \
//       -o /tmp/test_aclnn_ops && /tmp/test_aclnn_ops

#include "Runtime/AclnnOps.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace mlir::runtime::aclnn;

static int g_pass = 0, g_fail = 0;

static void checkClose(const float *got, const std::vector<float> &want,
                       float tol, const char *what) {
  bool ok = true;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::isnan(got[i]) || std::fabs(got[i] - want[i]) > tol) {
      ok = false;
      break;
    }
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("  FAIL: %s\n    got: ", what);
    for (size_t i = 0; i < want.size(); ++i) std::printf("%.5f ", got[i]);
    std::printf("\n    want:");
    for (float w : want) std::printf("%.5f ", w);
    std::printf("\n");
  }
}

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

// --- run_Matmul ------------------------------------------------------------
static void testMatmul2x2() {
  float a[] = {1, 2, 3, 4};            // [[1,2],[3,4]]
  float b[] = {5, 6, 7, 8};            // [[5,6],[7,8]]
  TensorInfo at = mk({2, 2}, a), bt = mk({2, 2}, b), out;
  run_Matmul(at, bt, /*init=*/at, &out, nullptr);
  // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
  checkClose((const float *)out.data, {19, 22, 43, 50}, 1e-4f, "matmul 2x2");
  freeTensor(&out);
}

static void testMatmulBatched() {
  // batch=2, [2,2,2]; second batch is the first scaled by 10.
  float a[] = {1, 2, 3, 4, 10, 20, 30, 40};
  float b[] = {5, 6, 7, 8, 5, 6, 7, 8};
  TensorInfo at = mk({2, 2, 2}, a), bt = mk({2, 2, 2}, b), out;
  run_Matmul(at, bt, at, &out, nullptr);
  checkClose((const float *)out.data,
             {19, 22, 43, 50, 190, 220, 430, 500}, 1e-3f, "matmul batched");
  freeTensor(&out);
}

// --- run_LayerNorm ---------------------------------------------------------
static void testLayerNorm() {
  float x[] = {1, 2, 3, 4};            // 1 row, D=4
  float gamma[] = {1, 1, 1, 1};
  float beta[] = {0, 0, 0, 0};
  TensorInfo xt = mk({1, 4}, x), gt = mk({4}, gamma), bt = mk({4}, beta), out;
  run_LayerNorm(xt, gt, bt, &out, nullptr);
  // mean=2.5, var=1.25, rstd=1/sqrt(1.25+1e-5)=0.894423
  // (x-mean)*rstd = [-1.5,-0.5,0.5,1.5]*0.894423
  checkClose((const float *)out.data,
             {-1.341635f, -0.447212f, 0.447212f, 1.341635f}, 1e-3f,
             "layernorm unit gamma/beta");
  freeTensor(&out);
}

static void testLayerNormAffine() {
  float x[] = {1, 2, 3, 4};
  float gamma[] = {2, 2, 2, 2};
  float beta[] = {1, 1, 1, 1};
  TensorInfo xt = mk({1, 4}, x), gt = mk({4}, gamma), bt = mk({4}, beta), out;
  run_LayerNorm(xt, gt, bt, &out, nullptr);
  // 2*norm + 1
  checkClose((const float *)out.data,
             {-1.683270f, 0.105576f, 1.894424f, 3.683270f}, 1e-3f,
             "layernorm affine");
  freeTensor(&out);
}

// --- run_Transpose ---------------------------------------------------------
static void testTranspose2D() {
  float in[] = {1, 2, 3, 4, 5, 6};     // 2x3 [[1,2,3],[4,5,6]]
  int64_t perm[] = {1, 0};
  TensorInfo it = mk({2, 3}, in), out;
  run_Transpose(it, perm, 2, &out, nullptr);
  // 3x2 [[1,4],[2,5],[3,6]]
  checkClose((const float *)out.data, {1, 4, 2, 5, 3, 6}, 1e-6f,
             "transpose 2x3 -> 3x2");
  freeTensor(&out);
}

static void testTranspose3D() {
  // [2,1,3] perm[1,0,2] -> [1,2,3] (just reorders the two leading dims).
  float in[] = {1, 2, 3, 4, 5, 6};
  int64_t perm[] = {1, 0, 2};
  TensorInfo it = mk({2, 1, 3}, in), out;
  run_Transpose(it, perm, 3, &out, nullptr);
  checkClose((const float *)out.data, {1, 2, 3, 4, 5, 6}, 1e-6f,
             "transpose [2,1,3] perm[1,0,2]");
  freeTensor(&out);
}

// --- run_FlashAttentionScore ----------------------------------------------
static void testFlashAttention() {
  // B=1, N=1, S=2, D=2.  Q=K=I so scores are symmetric; V=[[1,2],[3,4]].
  float q[] = {1, 0, 0, 1};
  float k[] = {1, 0, 0, 1};
  float v[] = {1, 2, 3, 4};
  TensorInfo qt = mk({1, 1, 2, 2}, q), kt = mk({1, 1, 2, 2}, k),
             vt = mk({1, 1, 2, 2}, v), out;
  // mask/init unused by the host reference; reuse q.
  run_FlashAttentionScore(qt, kt, vt, qt, qt, &out, nullptr);
  // scale=1/sqrt(2); softmax([s,0]) with s=0.70710678 -> self 0.669770,
  // other 0.330230.  out_row = w_self*V_self + w_other*V_other.
  checkClose((const float *)out.data,
             {1.66046f, 2.66046f, 2.33954f, 3.33954f}, 2e-3f,
             "flash-attention 1x1x2x2");
  freeTensor(&out);
}

int main() {
  setHostMode(true);  // no NPU here → exercise the CPU references

  testMatmul2x2();
  testMatmulBatched();
  testLayerNorm();
  testLayerNormAffine();
  testTranspose2D();
  testTranspose3D();
  testFlashAttention();

  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------
// DEVICE NOTE (real-NPU validation — run by the hardware session on a 910C):
//
//   c++ -std=c++17 -I include -I $ASCEND_HOME/<arch>/include \
//       test/tools/runtime/test_aclnn_ops.cpp lib/Runtime/AclnnOps.cpp \
//       -L $ASCEND_HOME/<arch>/lib64 -lascendcl -lnnopbase \
//       -lopapi_transformer ... -o /tmp/test_aclnn_ops_npu
//
// To exercise the aclnn (device) path, drop setHostMode(true) and aclInit +
// aclrtSetDevice first, then feed device buffers. The expected values above are
// backend-agnostic, so the device dispatch is validated against the same
// numbers (within a looser tol for f16 cube accumulation).
// ---------------------------------------------------------------------------
