#include "Runtime/AclnnOps.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// CANN headers — present on Ascend hosts; stubbed out for CI / dev builds.
// acl/acl_rt.h is included transitively via AclnnOps.h when CANN is available,
// so only the aclnn-op-specific header needs to be detected here.
#ifdef __has_include
#  if __has_include("aclnnop/aclnn_flash_attention_score.h")
#    include "aclnn/acl_meta.h"
#    include "aclnnop/aclnn_flash_attention_score.h"
#    include "aclnnop/aclnn_matmul.h"
#    include "aclnnop/aclnn_batch_matmul.h"
#    include "aclnnop/aclnn_layer_norm.h"
#    include "aclnnop/aclnn_permute.h"
#    define HAVE_CANN 1
#  endif
#endif

// ---------------------------------------------------------------------------
// Stubs — compile without CANN (CI, dev laptops, etc.)
// ---------------------------------------------------------------------------
#ifndef HAVE_CANN

using aclTensor      = void;
using aclOpExecutor  = void;
using aclDataType    = int;
using aclFormat      = int;
using aclIntArray    = void;

static constexpr int ACL_FLOAT          = 0;
static constexpr int ACL_FLOAT16        = 1;
static constexpr int ACL_FORMAT_ND      = 2;
static constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;

static void aclrtMallocStub(void **p, size_t sz, int) { *p = ::operator new(sz); }
static void aclrtFreeStub(void *p)                    { ::operator delete(p); }
#define aclrtMalloc(p, sz, f) aclrtMallocStub(p, sz, f)
#define aclrtFree(p)          aclrtFreeStub(p)

static constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
static constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
static int aclrtMemcpyStub(void *d, size_t, const void *s, size_t n, int) {
  std::memcpy(d, s, n);
  return 0;
}
#define aclrtMemcpy(d, dm, s, n, k) aclrtMemcpyStub((d), (dm), (s), (n), (k))

static aclTensor *aclCreateTensorStub(...) { return nullptr; }
static void aclDestroyTensorStub(const aclTensor *) {}
#define aclCreateTensor(...)   aclCreateTensorStub(__VA_ARGS__)
#define aclDestroyTensor(t)    aclDestroyTensorStub(t)

static aclIntArray *aclCreateIntArrayStub(const int64_t *, uint64_t) { return nullptr; }
static int aclDestroyIntArrayStub(const aclIntArray *) { return 0; }
#define aclCreateIntArray(v, n) aclCreateIntArrayStub((v), (n))
#define aclDestroyIntArray(a)   aclDestroyIntArrayStub(a)

// Generic no-op stubs for the matmul/layernorm/permute aclnn ops — used only on
// builds without CANN (CI/dev), where g_host_mode is always true so the device
// branch never actually runs.
static int aclnnGenericStub(...) { return 0; }
#define aclnnMatmulGetWorkspaceSize(...)      aclnnGenericStub(__VA_ARGS__)
#define aclnnMatmul(...)                      aclnnGenericStub(__VA_ARGS__)
#define aclnnBatchMatMulGetWorkspaceSize(...) aclnnGenericStub(__VA_ARGS__)
#define aclnnBatchMatMul(...)                 aclnnGenericStub(__VA_ARGS__)
#define aclnnLayerNormGetWorkspaceSize(...)   aclnnGenericStub(__VA_ARGS__)
#define aclnnLayerNorm(...)                   aclnnGenericStub(__VA_ARGS__)
#define aclnnPermuteGetWorkspaceSize(...)     aclnnGenericStub(__VA_ARGS__)
#define aclnnPermute(...)                     aclnnGenericStub(__VA_ARGS__)

// CANN 9.0.0 aclnnFlashAttentionScoreGetWorkspaceSize signature (22 params):
//   query, key, value,
//   realShiftOpt, dropMaskOpt, paddingMaskOpt, attenMaskOpt, prefixOpt,
//   scaleValue, keepProb, preTokens, nextTokens,
//   headNum, inputLayout, innerPrecise, sparseMode,
//   softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
//   workspaceSize, executor
static int aclnnFlashAttentionScoreGetWorkspaceSizeStub(
    aclTensor *, aclTensor *, aclTensor *,
    aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclIntArray *,
    double, double, int64_t, int64_t,
    int64_t, char *, int64_t, int64_t,
    aclTensor *, aclTensor *, aclTensor *, aclTensor *,
    uint64_t *, aclOpExecutor **) { return 0; }
static int aclnnFlashAttentionScoreStub(void *, uint64_t, aclOpExecutor *, void *) { return 0; }

#define aclnnFlashAttentionScoreGetWorkspaceSize(...) \
    aclnnFlashAttentionScoreGetWorkspaceSizeStub(__VA_ARGS__)
#define aclnnFlashAttentionScore(...) aclnnFlashAttentionScoreStub(__VA_ARGS__)

#endif // !HAVE_CANN

namespace mlir::runtime::aclnn {

// ---------------------------------------------------------------------------
// Host-mode flag — set when aclInit fails (no NPU hardware available)
// ---------------------------------------------------------------------------
static bool g_host_mode = false;
void setHostMode(bool e) { g_host_mode = e; }
bool isHostMode()        { return g_host_mode; }

// ---------------------------------------------------------------------------
// aclDataType → bytes
// ---------------------------------------------------------------------------
static size_t elemBytes(int dtype) {
  switch (dtype) {
    case 0:  return 4;  // ACL_FLOAT   = float32
    case 1:  return 2;  // ACL_FLOAT16 = float16
    case 2:  return 1;  // ACL_INT8
    case 3:  return 4;  // ACL_INT32
    case 27: return 2;  // ACL_BF16
    default: return 2;
  }
}

void rowMajorStrides(const int64_t *shape, int rank, int64_t *strides) {
  if (rank == 0) return;
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
}

void allocTensorLike(const TensorInfo &src, TensorInfo *dst) {
  assert(src.rank > 0 && src.rank <= 8);
  dst->rank  = src.rank;
  dst->dtype = src.dtype;
  std::memcpy(dst->shape, src.shape, src.rank * sizeof(int64_t));
  rowMajorStrides(dst->shape, dst->rank, dst->strides);

  size_t n = 1;
  for (int i = 0; i < src.rank; ++i)
    n *= static_cast<size_t>(src.shape[i]);
  size_t nb = n * elemBytes(src.dtype);
  if (g_host_mode)
    dst->data = ::operator new(nb);
  else
    aclrtMalloc(&dst->data, nb, ACL_MEM_MALLOC_NORMAL_ONLY);
}

void freeTensor(TensorInfo *t) {
  if (t && t->data) {
    if (g_host_mode)
      ::operator delete(t->data);
    else
      aclrtFree(t->data);
    t->data = nullptr;
  }
}

// ---------------------------------------------------------------------------
// Device staging (backend=npu).  The generated network_host.cpp orchestrates
// every tensor in HOST memory (::operator new) and the AscendC launch path
// (hostLaunchAscendCKernel) is host-in/host-out — it stages its own H2D/D2H
// internally.  So the aclnn wrappers must follow the same convention: treat
// TensorInfo.data as host pointers, copy inputs H2D into temporary device
// buffers around the aclnn call, and copy the result D2H back into a fresh host
// buffer.  Mixing the two domains (host ptr read as device, or vice-versa) is
// what broke the integrated phase-5 run.
// [PERF FUTURE: unify on the device domain instead — make host-gen aclrtMalloc
//  all intermediates and hostLaunchAscendCKernel device-pointer-aware — to drop
//  these per-op round-trips.  See 2026-05-22-real-npu-aclnn-direct-handoff.md.]
// ---------------------------------------------------------------------------
static size_t tensorBytes(const TensorInfo &t) {
  size_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= static_cast<size_t>(t.shape[i]);
  return n * elemBytes(t.dtype);
}

// Copy a host-resident TensorInfo into a fresh device buffer; return the
// device-backed descriptor and record the device ptr in `pool` for freeing.
static TensorInfo stageToDevice(const TensorInfo &hostT,
                                std::vector<void *> &pool) {
  TensorInfo d = hostT;
  rowMajorStrides(d.shape, d.rank, d.strides);
  size_t nb = tensorBytes(hostT);
  aclrtMalloc(&d.data, nb, ACL_MEM_MALLOC_NORMAL_ONLY);
  aclrtMemcpy(d.data, nb, hostT.data, nb, ACL_MEMCPY_HOST_TO_DEVICE);
  pool.push_back(d.data);
  return d;
}

// Allocate a device output buffer shaped like `tmpl`; record ptr in `pool`.
static TensorInfo stageDeviceOut(const TensorInfo &tmpl,
                                 std::vector<void *> &pool) {
  TensorInfo d = tmpl;
  rowMajorStrides(d.shape, d.rank, d.strides);
  aclrtMalloc(&d.data, tensorBytes(tmpl), ACL_MEM_MALLOC_NORMAL_ONLY);
  pool.push_back(d.data);
  return d;
}

// After the aclnn op wrote `devOut`, copy it D2H into a fresh HOST buffer so the
// downstream host-domain consumers (AscendC launch / operator-new chain) see it.
static void stageToHost(const TensorInfo &devOut, TensorInfo *out) {
  *out = devOut;
  size_t nb = tensorBytes(devOut);
  out->data = ::operator new(nb);
  aclrtMemcpy(out->data, nb, devOut.data, nb, ACL_MEMCPY_DEVICE_TO_HOST);
}

// Free every device buffer recorded in `pool`.
static void freePool(std::vector<void *> &pool) {
  for (void *p : pool)
    if (p) aclrtFree(p);
}

// ---------------------------------------------------------------------------
// CPU reference: scaled dot-product attention for BNSD tensors (float16 I/O).
// out is allocated with operator new; caller must free via freeTensor().
// ---------------------------------------------------------------------------
static float h2f(uint16_t h) {
  uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff, f;
  if (exp == 0)       f = (sign << 31) | (frac << 13);
  else if (exp == 31) f = (sign << 31) | 0x7f800000u | (frac << 13);
  else                f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
  float v; std::memcpy(&v, &f, 4); return v;
}
static uint16_t f2h(float fv) {
  uint32_t x; std::memcpy(&x, &fv, 4);
  uint16_t sign = (x >> 16) & 0x8000;
  int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 112;
  uint32_t frac = x & 0x7fffff;
  if (exp <= 0)  return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (uint16_t)(exp << 10) | (uint16_t)(frac >> 13);
}

static void sdpa_cpu(const TensorInfo &q, const TensorInfo &k,
                     const TensorInfo &v, TensorInfo *out) {
  int64_t B = q.shape[0], N = q.shape[1], S = q.shape[2], D = q.shape[3];
  float scale = 1.0f / std::sqrt((float)D);

  allocTensorLike(q, out);  // allocates host memory when g_host_mode

  auto idx = [&](int64_t b, int64_t n, int64_t s, int64_t d) {
    return (size_t)((b * N + n) * S * D + s * D + d);
  };
  // dtype-agnostic element access: 1 = ACL_FLOAT16, 0 = ACL_FLOAT (f32).
  bool f16 = (q.dtype == 1);
  auto rd = [&](const void *p, size_t i) -> float {
    return f16 ? h2f(((const uint16_t *)p)[i]) : ((const float *)p)[i];
  };
  auto wr = [&](void *p, size_t i, float val) {
    if (f16) ((uint16_t *)p)[i] = f2h(val);
    else ((float *)p)[i] = val;
  };

  std::vector<float> scores((size_t)(S * S));

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      // QK^T / sqrt(D)
      for (int64_t s1 = 0; s1 < S; ++s1)
        for (int64_t s2 = 0; s2 < S; ++s2) {
          float dot = 0.f;
          for (int64_t d = 0; d < D; ++d)
            dot += rd(q.data, idx(b, n, s1, d)) * rd(k.data, idx(b, n, s2, d));
          scores[(size_t)(s1 * S + s2)] = dot * scale;
        }
      // softmax row-wise
      for (int64_t s1 = 0; s1 < S; ++s1) {
        float mx = scores[(size_t)(s1 * S)];
        for (int64_t s2 = 1; s2 < S; ++s2) mx = std::max(mx, scores[(size_t)(s1 * S + s2)]);
        float sum = 0.f;
        for (int64_t s2 = 0; s2 < S; ++s2) { scores[(size_t)(s1*S+s2)] = std::exp(scores[(size_t)(s1*S+s2)] - mx); sum += scores[(size_t)(s1*S+s2)]; }
        for (int64_t s2 = 0; s2 < S; ++s2) scores[(size_t)(s1*S+s2)] /= sum;
      }
      // out = scores @ V
      for (int64_t s = 0; s < S; ++s)
        for (int64_t d = 0; d < D; ++d) {
          float acc = 0.f;
          for (int64_t k2 = 0; k2 < S; ++k2)
            acc += scores[(size_t)(s * S + k2)] * rd(v.data, idx(b, n, k2, d));
          wr(out->data, idx(b, n, s, d), acc);
        }
    }
  }
}

// ---------------------------------------------------------------------------
// Wrap a TensorInfo into an aclTensor (row-major strides, ND format).
// Caller must release via aclDestroyTensor().
// ---------------------------------------------------------------------------
static aclTensor *makeAclTensor(const TensorInfo &ti) {
  return aclCreateTensor(
      ti.shape,   static_cast<uint64_t>(ti.rank),
      static_cast<aclDataType>(ti.dtype),
      ti.strides, /*storageOffset=*/0,
      static_cast<aclFormat>(ACL_FORMAT_ND),
      ti.shape,   static_cast<uint64_t>(ti.rank),
      ti.data);
}

// ---------------------------------------------------------------------------
// run_FlashAttentionScore
//
// Calls aclnnFlashAttentionScore (CANN 9.0.0) for bidirectional attention.
// Signature of aclnnFlashAttentionScoreGetWorkspaceSize:
//   (query, key, value,
//    realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
//    prefixOptional,
//    scaleValue, keepProb, preTokens, nextTokens,
//    headNum, inputLayout, innerPrecise, sparseMode,
//    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
//    workspaceSize, executor)
// ---------------------------------------------------------------------------
void run_FlashAttentionScore(
    TensorInfo q, TensorInfo k, TensorInfo v,
    TensorInfo /*mask*/, TensorInfo /*init*/,
    TensorInfo *out, aclrtStream stream) {

  assert(q.rank == 4 && "expected BNSD rank-4 query");

  if (g_host_mode) {
    sdpa_cpu(q, k, v, out);
    return;
  }

  int64_t numHeads = q.shape[1];
  int64_t headDim  = q.shape[3];
  double  scale    = 1.0 / std::sqrt(static_cast<double>(headDim));

  // host-in/host-out: stage q/k/v H2D, run on device, copy result D2H.
  std::vector<void *> pool;
  TensorInfo qD = stageToDevice(q, pool);
  TensorInfo kD = stageToDevice(k, pool);
  TensorInfo vD = stageToDevice(v, pool);
  TensorInfo oD = stageDeviceOut(q, pool);  // output shape == q

  aclTensor *qT   = makeAclTensor(qD);
  aclTensor *kT   = makeAclTensor(kD);
  aclTensor *vT   = makeAclTensor(vD);
  aclTensor *outT = makeAclTensor(oD);

  // Allocate softmax intermediates required by the training-oriented API.
  // Shape: [B, N, S, 8] float32 (8-element alignment used by flash attention).
  TensorInfo softmaxMax, softmaxSum;
  {
    int64_t auxShape[4] = {q.shape[0], q.shape[1], q.shape[2], 8};
    softmaxMax.rank  = softmaxSum.rank  = 4;
    softmaxMax.dtype = softmaxSum.dtype = ACL_FLOAT;  // float32
    std::memcpy(softmaxMax.shape, auxShape, 4 * sizeof(int64_t));
    std::memcpy(softmaxSum.shape, auxShape, 4 * sizeof(int64_t));
    softmaxMax = stageDeviceOut(softmaxMax, pool);
    softmaxSum = stageDeviceOut(softmaxSum, pool);
  }
  aclTensor *softmaxMaxT = makeAclTensor(softmaxMax);
  aclTensor *softmaxSumT = makeAclTensor(softmaxSum);

  uint64_t      wsSize   = 0;
  aclOpExecutor *executor = nullptr;

  int rc = aclnnFlashAttentionScoreGetWorkspaceSize(
      qT, kT, vT,
      /*realShiftOptional=*/nullptr,    // no positional bias
      /*dropMaskOptional=*/nullptr,     // no dropout
      /*paddingMaskOptional=*/nullptr,  // no padding mask
      /*attenMaskOptional=*/nullptr,    // full bidirectional attention
      /*prefixOptional=*/nullptr,
      scale,
      /*keepProb=*/1.0,
      /*preTokens=*/65536,             // attend to all previous tokens
      /*nextTokens=*/65536,            // attend to all following tokens
      numHeads,
      const_cast<char *>("BNSD"),
      /*innerPrecise=*/0,
      /*sparseMode=*/0,
      softmaxMaxT, softmaxSumT,
      /*softmaxOutOut=*/nullptr,        // not needed for forward-only
      outT,
      &wsSize, &executor);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] aclnnFlashAttentionScoreGetWorkspaceSize rc=%d\n", rc);

  void *ws = nullptr;
  if (wsSize > 0)
    aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);

  rc = aclnnFlashAttentionScore(ws, wsSize, executor, stream);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] aclnnFlashAttentionScore rc=%d\n", rc);

  if (ws) aclrtFree(ws);
  stageToHost(oD, out);  // D2H into a fresh host buffer

  aclDestroyTensor(qT);
  aclDestroyTensor(kT);
  aclDestroyTensor(vT);
  aclDestroyTensor(outT);
  aclDestroyTensor(softmaxMaxT);
  aclDestroyTensor(softmaxSumT);
  freePool(pool);
}

// ---------------------------------------------------------------------------
// CPU reference: (batched) matrix multiply.  a:[..,M,K] b:[..,K,N] -> [..,M,N].
// f16/f32, row-major.  Used as the aclnn fallback for cube (matmul) groups the
// AscendC codegen can't yet handle.
// ---------------------------------------------------------------------------
static void matmul_cpu(const TensorInfo &a, const TensorInfo &b,
                       TensorInfo *out) {
  int rank = a.rank;
  assert(rank >= 2 && b.rank == rank && "matmul expects matching rank >= 2");
  int64_t M = a.shape[rank - 2], K = a.shape[rank - 1];
  int64_t N = b.shape[rank - 1];
  assert(b.shape[rank - 2] == K && "matmul inner dim mismatch");
  int64_t batch = 1;
  for (int i = 0; i < rank - 2; ++i)
    batch *= a.shape[i];

  // out template: a's shape with last dim -> N.
  TensorInfo tmpl = a;
  tmpl.shape[rank - 1] = N;
  allocTensorLike(tmpl, out);

  bool f16 = (a.dtype == 1);
  auto rd = [&](const void *p, size_t i) -> float {
    return f16 ? h2f(((const uint16_t *)p)[i]) : ((const float *)p)[i];
  };
  auto wr = [&](void *p, size_t i, float v) {
    if (f16) ((uint16_t *)p)[i] = f2h(v);
    else ((float *)p)[i] = v;
  };
  for (int64_t bi = 0; bi < batch; ++bi)
    for (int64_t m = 0; m < M; ++m)
      for (int64_t n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int64_t k = 0; k < K; ++k)
          acc += rd(a.data, (size_t)((bi * M + m) * K + k)) *
                 rd(b.data, (size_t)((bi * K + k) * N + n));
        wr(out->data, (size_t)((bi * M + m) * N + n), acc);
      }
}

void run_Matmul(TensorInfo a, TensorInfo b, TensorInfo /*init*/,
                TensorInfo *out, aclrtStream stream) {
  if (g_host_mode) {
    matmul_cpu(a, b, out);
    return;
  }

  // host-in/host-out staging.
  std::vector<void *> pool;
  TensorInfo aD = stageToDevice(a, pool);
  TensorInfo bD = stageToDevice(b, pool);
  // out template: a's shape with last dim -> N (= b's last dim).
  TensorInfo tmpl = a;
  tmpl.shape[a.rank - 1] = b.shape[b.rank - 1];
  TensorInfo oD = stageDeviceOut(tmpl, pool);

  aclTensor *aT = makeAclTensor(aD);
  aclTensor *bT = makeAclTensor(bD);
  aclTensor *oT = makeAclTensor(oD);

  // cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION): fp32 inputs run on the fp16
  // cube. This is the device-accuracy knob the hardware session may tune
  // (0 = KEEP_DTYPE) if fp32 precision is required.
  const int8_t cubeMathType = 1;
  bool batched = (a.rank > 2);
  uint64_t       wsSize   = 0;
  aclOpExecutor *executor = nullptr;
  int rc = batched
    ? aclnnBatchMatMulGetWorkspaceSize(aT, bT, oT, cubeMathType, &wsSize, &executor)
    : aclnnMatmulGetWorkspaceSize(aT, bT, oT, cubeMathType, &wsSize, &executor);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] %sMatMulGetWorkspaceSize rc=%d\n",
            batched ? "Batch" : "", rc);

  void *ws = nullptr;
  if (wsSize > 0)
    aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);

  rc = batched ? aclnnBatchMatMul(ws, wsSize, executor, stream)
               : aclnnMatmul(ws, wsSize, executor, stream);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] %sMatMul rc=%d\n", batched ? "Batch" : "", rc);

  if (ws) aclrtFree(ws);
  stageToHost(oD, out);
  aclDestroyTensor(aT);
  aclDestroyTensor(bT);
  aclDestroyTensor(oT);
  freePool(pool);
}

// ---------------------------------------------------------------------------
// 2D convolution CPU reference (NCHW input × FCHW weight → NCHW output).
// Padding is materialized upstream as tensor.pad — the kernel sees the already
// padded input, so we apply only strides and dilations here.
// ---------------------------------------------------------------------------
static void conv2d_cpu(const TensorInfo &in, const TensorInfo &weight,
                       const int64_t *strides, const int64_t *dilations,
                       TensorInfo *out) {
  assert(in.rank == 4 && weight.rank == 4 && "conv2d expects NCHW × FCHW");
  int64_t N = in.shape[0], C = in.shape[1], H = in.shape[2], W = in.shape[3];
  int64_t F = weight.shape[0], KH = weight.shape[2], KW = weight.shape[3];
  assert(weight.shape[1] == C && "conv2d in-channel mismatch");
  int64_t SH = strides[0], SW = strides[1];
  int64_t DH = dilations[0], DW = dilations[1];
  int64_t OH = (H - DH * (KH - 1) - 1) / SH + 1;
  int64_t OW = (W - DW * (KW - 1) - 1) / SW + 1;

  TensorInfo tmpl = in;
  tmpl.shape[0] = N; tmpl.shape[1] = F; tmpl.shape[2] = OH; tmpl.shape[3] = OW;
  allocTensorLike(tmpl, out);

  bool f16 = (in.dtype == 1);
  auto rd = [&](const void *p, size_t i) -> float {
    return f16 ? h2f(((const uint16_t *)p)[i]) : ((const float *)p)[i];
  };
  auto wr = [&](void *p, size_t i, float v) {
    if (f16) ((uint16_t *)p)[i] = f2h(v);
    else ((float *)p)[i] = v;
  };

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t f = 0; f < F; ++f) {
      for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
          float acc = 0.f;
          for (int64_t c = 0; c < C; ++c) {
            for (int64_t kh = 0; kh < KH; ++kh) {
              int64_t ih = oh * SH + kh * DH;
              for (int64_t kw = 0; kw < KW; ++kw) {
                int64_t iw = ow * SW + kw * DW;
                size_t iidx = (size_t)(((n * C + c) * H + ih) * W + iw);
                size_t widx = (size_t)(((f * C + c) * KH + kh) * KW + kw);
                acc += rd(in.data, iidx) * rd(weight.data, widx);
              }
            }
          }
          size_t oidx = (size_t)(((n * F + f) * OH + oh) * OW + ow);
          wr(out->data, oidx, acc);
        }
      }
    }
  }
}

void run_Conv2D(TensorInfo in, TensorInfo weight, TensorInfo /*init*/,
                const int64_t *strides, const int64_t *dilations,
                TensorInfo *out, aclrtStream /*stream*/) {
  if (!g_host_mode) {
    fprintf(stderr, "[AclnnOps] run_Conv2D: device path not implemented, "
                    "running CPU reference on host buffers\n");
  }
  conv2d_cpu(in, weight, strides, dilations, out);
}

// ---------------------------------------------------------------------------
// LayerNorm CPU reference: normalize over the last dim (size D = gamma.shape).
//   out = (x - mean) / sqrt(var + eps) * gamma + beta
// eps is the torch default; biased variance (divide by D), matching nn.LayerNorm.
// ---------------------------------------------------------------------------
static void layernorm_cpu(const TensorInfo &x, const TensorInfo &gamma,
                          const TensorInfo &beta, TensorInfo *out) {
  const float eps = 1e-5f;
  int64_t D = gamma.shape[0];
  int64_t rows = 1;
  for (int i = 0; i < x.rank - 1; ++i)
    rows *= x.shape[i];

  allocTensorLike(x, out);

  bool f16 = (x.dtype == 1);
  auto rd = [&](const void *p, size_t i) -> float {
    return f16 ? h2f(((const uint16_t *)p)[i]) : ((const float *)p)[i];
  };
  auto wr = [&](void *p, size_t i, float v) {
    if (f16) ((uint16_t *)p)[i] = f2h(v);
    else ((float *)p)[i] = v;
  };

  for (int64_t r = 0; r < rows; ++r) {
    size_t base = (size_t)(r * D);
    float mean = 0.f;
    for (int64_t d = 0; d < D; ++d)
      mean += rd(x.data, base + d);
    mean /= (float)D;
    float var = 0.f;
    for (int64_t d = 0; d < D; ++d) {
      float c = rd(x.data, base + d) - mean;
      var += c * c;
    }
    var /= (float)D;
    float rstd = 1.0f / std::sqrt(var + eps);
    for (int64_t d = 0; d < D; ++d) {
      float norm = (rd(x.data, base + d) - mean) * rstd;
      wr(out->data, base + d,
         norm * rd(gamma.data, (size_t)d) + rd(beta.data, (size_t)d));
    }
  }
}

void run_LayerNorm(TensorInfo x, TensorInfo gamma, TensorInfo beta,
                   TensorInfo *out, aclrtStream stream) {
  if (g_host_mode) {
    layernorm_cpu(x, gamma, beta, out);
    return;
  }

  // host-in/host-out staging.
  std::vector<void *> pool;
  TensorInfo xD = stageToDevice(x, pool);
  TensorInfo gD = stageToDevice(gamma, pool);
  TensorInfo bD = stageToDevice(beta, pool);
  TensorInfo oD = stageDeviceOut(x, pool);  // output shape == x

  int64_t D = gamma.shape[0];  // normalized over the last dim
  aclIntArray *normShape = aclCreateIntArray(&D, 1);

  aclTensor *xT = makeAclTensor(xD);
  aclTensor *gT = makeAclTensor(gD);
  aclTensor *bT = makeAclTensor(bD);
  aclTensor *oT = makeAclTensor(oD);

  // mean/rstd aux outputs: x's shape with the normalized (last) dim removed.
  TensorInfo meanI, rstdI;
  int auxRank = x.rank > 1 ? x.rank - 1 : 1;
  meanI.rank = rstdI.rank = auxRank;
  for (int i = 0; i < auxRank; ++i) {
    int64_t d = (x.rank > 1) ? x.shape[i] : 1;
    meanI.shape[i] = rstdI.shape[i] = d;
  }
  meanI.dtype = rstdI.dtype = ACL_FLOAT;  // mean/rstd are always fp32
  meanI = stageDeviceOut(meanI, pool);
  rstdI = stageDeviceOut(rstdI, pool);
  aclTensor *meanT = makeAclTensor(meanI);
  aclTensor *rstdT = makeAclTensor(rstdI);

  uint64_t       wsSize   = 0;
  aclOpExecutor *executor = nullptr;
  int rc = aclnnLayerNormGetWorkspaceSize(
      xT, normShape, gT, bT, /*eps=*/1e-5, oT, meanT, rstdT, &wsSize, &executor);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] aclnnLayerNormGetWorkspaceSize rc=%d\n", rc);

  void *ws = nullptr;
  if (wsSize > 0)
    aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);

  rc = aclnnLayerNorm(ws, wsSize, executor, stream);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] aclnnLayerNorm rc=%d\n", rc);

  if (ws) aclrtFree(ws);
  stageToHost(oD, out);
  aclDestroyIntArray(normShape);
  aclDestroyTensor(xT);
  aclDestroyTensor(gT);
  aclDestroyTensor(bT);
  aclDestroyTensor(oT);
  aclDestroyTensor(meanT);
  aclDestroyTensor(rstdT);
  freePool(pool);
}

// ---------------------------------------------------------------------------
// Transpose CPU reference: out[i] = in[j], j[perm[d]] = i[d].
//   out.shape[d] = in.shape[perm[d]]
// ---------------------------------------------------------------------------
static void transpose_cpu(const TensorInfo &in, const int64_t *perm, int rank,
                          TensorInfo *out) {
  TensorInfo tmpl = in;
  tmpl.rank = rank;
  for (int d = 0; d < rank; ++d)
    tmpl.shape[d] = in.shape[perm[d]];
  allocTensorLike(tmpl, out);

  int64_t inStride[8];
  inStride[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; --d)
    inStride[d] = inStride[d + 1] * in.shape[d + 1];

  int64_t total = 1;
  for (int d = 0; d < rank; ++d)
    total *= tmpl.shape[d];

  bool f16 = (in.dtype == 1);
  auto rd = [&](size_t i) -> float {
    return f16 ? h2f(((const uint16_t *)in.data)[i])
               : ((const float *)in.data)[i];
  };
  auto wr = [&](size_t i, float v) {
    if (f16) ((uint16_t *)out->data)[i] = f2h(v);
    else ((float *)out->data)[i] = v;
  };

  int64_t idx[8] = {0};
  for (int64_t o = 0; o < total; ++o) {
    size_t s = 0;
    for (int d = 0; d < rank; ++d)
      s += (size_t)idx[d] * (size_t)inStride[perm[d]];
    wr((size_t)o, rd(s));
    for (int d = rank - 1; d >= 0; --d) {
      if (++idx[d] < tmpl.shape[d])
        break;
      idx[d] = 0;
    }
  }
}

void run_Transpose(TensorInfo in, const int64_t *perm, int rank,
                   TensorInfo *out, aclrtStream stream) {
  if (g_host_mode) {
    transpose_cpu(in, perm, rank, out);
    return;
  }

  // host-in/host-out staging.
  std::vector<void *> pool;
  TensorInfo inD = stageToDevice(in, pool);
  // out template: out.shape[d] = in.shape[perm[d]].
  TensorInfo tmpl = in;
  tmpl.rank = rank;
  for (int d = 0; d < rank; ++d)
    tmpl.shape[d] = in.shape[perm[d]];
  TensorInfo oD = stageDeviceOut(tmpl, pool);

  aclTensor *iT = makeAclTensor(inD);
  aclTensor *oT = makeAclTensor(oD);
  aclIntArray *dims = aclCreateIntArray(perm, static_cast<uint64_t>(rank));

  uint64_t       wsSize   = 0;
  aclOpExecutor *executor = nullptr;
  int rc = aclnnPermuteGetWorkspaceSize(iT, dims, oT, &wsSize, &executor);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] aclnnPermuteGetWorkspaceSize rc=%d\n", rc);

  void *ws = nullptr;
  if (wsSize > 0)
    aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);

  rc = aclnnPermute(ws, wsSize, executor, stream);
  if (rc != 0)
    fprintf(stderr, "[AclnnOps] aclnnPermute rc=%d\n", rc);

  if (ws) aclrtFree(ws);
  stageToHost(oD, out);
  aclDestroyIntArray(dims);
  aclDestroyTensor(iT);
  aclDestroyTensor(oT);
  freePool(pool);
}

} // namespace mlir::runtime::aclnn