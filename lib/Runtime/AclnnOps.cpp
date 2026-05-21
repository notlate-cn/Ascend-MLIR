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

static aclTensor *aclCreateTensorStub(...) { return nullptr; }
static void aclDestroyTensorStub(const aclTensor *) {}
#define aclCreateTensor(...)   aclCreateTensorStub(__VA_ARGS__)
#define aclDestroyTensor(t)    aclDestroyTensorStub(t)

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
  auto qf16 = (const uint16_t *)q.data;
  auto kf16 = (const uint16_t *)k.data;
  auto vf16 = (const uint16_t *)v.data;
  auto of16 = (uint16_t *)out->data;

  std::vector<float> scores((size_t)(S * S));

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      // QK^T / sqrt(D)
      for (int64_t s1 = 0; s1 < S; ++s1)
        for (int64_t s2 = 0; s2 < S; ++s2) {
          float dot = 0.f;
          for (int64_t d = 0; d < D; ++d)
            dot += h2f(qf16[idx(b, n, s1, d)]) * h2f(kf16[idx(b, n, s2, d)]);
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
            acc += scores[(size_t)(s * S + k2)] * h2f(vf16[idx(b, n, k2, d)]);
          of16[idx(b, n, s, d)] = f2h(acc);
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

  rowMajorStrides(q.shape, q.rank, q.strides);
  rowMajorStrides(k.shape, k.rank, k.strides);
  rowMajorStrides(v.shape, v.rank, v.strides);

  aclTensor *qT = makeAclTensor(q);
  aclTensor *kT = makeAclTensor(k);
  aclTensor *vT = makeAclTensor(v);

  // Allocate main attention output (same shape + dtype as q).
  allocTensorLike(q, out);
  aclTensor *outT = makeAclTensor(*out);

  // Allocate softmax intermediates required by the training-oriented API.
  // Shape: [B, N, S, 8] float32 (8-element alignment used by flash attention).
  TensorInfo softmaxMax, softmaxSum;
  {
    int64_t auxShape[4] = {q.shape[0], q.shape[1], q.shape[2], 8};
    softmaxMax.rank  = softmaxSum.rank  = 4;
    softmaxMax.dtype = softmaxSum.dtype = ACL_FLOAT;  // float32
    std::memcpy(softmaxMax.shape, auxShape, 4 * sizeof(int64_t));
    std::memcpy(softmaxSum.shape, auxShape, 4 * sizeof(int64_t));
    rowMajorStrides(softmaxMax.shape, 4, softmaxMax.strides);
    rowMajorStrides(softmaxSum.shape, 4, softmaxSum.strides);
    size_t auxBytes = static_cast<size_t>(
        auxShape[0] * auxShape[1] * auxShape[2] * 8) * sizeof(float);
    aclrtMalloc(&softmaxMax.data, auxBytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&softmaxSum.data, auxBytes, ACL_MEM_MALLOC_NORMAL_ONLY);
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

  aclDestroyTensor(qT);
  aclDestroyTensor(kT);
  aclDestroyTensor(vT);
  aclDestroyTensor(outT);
  aclDestroyTensor(softmaxMaxT);
  aclDestroyTensor(softmaxSumT);
  freeTensor(&softmaxMax);
  freeTensor(&softmaxSum);
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
                TensorInfo *out, aclrtStream /*stream*/) {
  assert(g_host_mode &&
         "run_Matmul: only the host-mode CPU reference is implemented");
  matmul_cpu(a, b, out);
}

} // namespace mlir::runtime::aclnn