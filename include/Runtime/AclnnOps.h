// include/Runtime/AclnnOps.h
// Runtime helpers for aclnn direct-call ops (FlashAttentionScore, …).
// Generated network_host.cpp #includes this header and calls the run_* wrappers.
#pragma once

#include <cstdint>

// Forward-declare CANN stream type for builds without CANN installed.
// CANN defines aclrtStream as "typedef void*", so void* is compatible.
#if defined(__has_include) && __has_include("acl/acl_rt.h")
#  include "acl/acl_rt.h"
#else
using aclrtStream = void *;
#endif

namespace mlir::runtime::aclnn {

// Flat tensor descriptor used throughout network_host.cpp.
// All shapes are row-major; strides are in element units.
struct TensorInfo {
  void    *data     = nullptr;
  int64_t  shape[8] = {};
  int64_t  strides[8] = {};
  int      rank     = 0;
  int      dtype    = 0;  // aclDataType value
};

// Fill strides[] with row-major strides derived from shape[0..rank-1].
void rowMajorStrides(const int64_t *shape, int rank, int64_t *strides);

// Allocate device memory sized for src and write result into *dst.
// dst->shape/rank/dtype are copied from src; dst->data is aclrtMalloc'd.
void allocTensorLike(const TensorInfo &src, TensorInfo *dst);

// Release memory previously allocated by allocTensorLike / run_*.
void freeTensor(TensorInfo *t);

// Enable host-only mode: tensors live in malloc'd host memory and ops run a
// CPU reference implementation.  Call before network() when aclInit fails.
void setHostMode(bool enabled);
bool isHostMode();

// -------------------------------------------------------------------
// aclnn wrappers — one per aclnn.op kind
// -------------------------------------------------------------------

// Wraps aclnnFlashAttentionScoreGetWorkspaceSize + aclnnFlashAttentionScore.
// Inputs are BNSD tensors (rank 4).  Output is allocated inside this call.
// Caller must eventually call freeTensor(out).
void run_FlashAttentionScore(
    TensorInfo q,    // [B, N, S, D]  query
    TensorInfo k,    // [B, N, S, D]  key
    TensorInfo v,    // [B, N, S, D]  value
    TensorInfo mask, // [B, N, S, S]  attention mask (additive, fp16)
    TensorInfo init, // [B, N, S, D]  init output (may be zeros; currently unused by CANN)
    TensorInfo *out, // written by this call
    aclrtStream stream);

// CPU-reference matrix multiply (aclnn-fallback for linalg.matmul /
// batch_matmul).  a: [.., M, K], b: [.., K, N] -> out: [.., M, N], leading dims
// are batch.  `init` is the DPS output buffer (unused).  Host-mode only.
void run_Matmul(
    TensorInfo a,
    TensorInfo b,
    TensorInfo init, // DPS init/output buffer (unused by the reference)
    TensorInfo *out, // allocated + written by this call
    aclrtStream stream);

// CPU-reference LayerNorm (aclnn-fallback for the decomposed nn.LayerNorm
// subgraph).  Normalizes over the last dim (size = gamma.shape[0]):
//   out = (x - mean) / sqrt(var + eps) * gamma + beta
// eps is the torch default (1e-5).  x: [.., D], gamma/beta: [D].  Host-mode only.
void run_LayerNorm(
    TensorInfo x,
    TensorInfo gamma,
    TensorInfo beta,
    TensorInfo *out, // allocated + written by this call
    aclrtStream stream);

} // namespace mlir::runtime::aclnn
