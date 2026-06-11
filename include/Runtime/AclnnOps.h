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
// are batch.  `init` is the DPS output buffer (unused).  Host mode: CPU
// reference; device: aclnnMatmul (rank 2) / aclnnBatchMatMul (rank > 2).
void run_Matmul(
    TensorInfo a,
    TensorInfo b,
    TensorInfo init, // DPS init/output buffer (unused by the reference)
    TensorInfo *out, // allocated + written by this call
    aclrtStream stream);

// CPU-reference LayerNorm (aclnn-fallback for the decomposed nn.LayerNorm
// subgraph).  Normalizes over the last dim (size = gamma.shape[0]):
//   out = (x - mean) / sqrt(var + eps) * gamma + beta
// eps is the torch default (1e-5).  x: [.., D], gamma/beta: [D].  Host mode: CPU
// reference; device: aclnnLayerNorm (normalized over the last dim).
void run_LayerNorm(
    TensorInfo x,
    TensorInfo gamma,
    TensorInfo beta,
    TensorInfo *out, // allocated + written by this call
    aclrtStream stream);

// CPU-reference Embedding lookup (aclnn-fallback for the gather pattern
// torch.export emits for nn.Embedding).  table: [V, F] (vocab x feature),
// indices: [..., N] (any rank, integer dtype int32 or int64).
// out:     [..., N, F]  (= indices.shape ++ [F]).
// Row lookup: out[..., i, :] = table[indices[..., i], :].  Host mode: CPU
// reference; device mode: not wired (falls back to CPU).
void run_Embedding(
    TensorInfo table,
    TensorInfo indices,
    TensorInfo *out,
    aclrtStream stream);

// CPU-reference BatchNorm-eval (aclnn-fallback for the decomposed
// nn.BatchNorm2d eval-mode subgraph).  Per-channel affine over [..., C, ...]
// where C = weight.shape[0]:
//   out = (x - running_mean) / sqrt(running_var + eps) * weight + bias
// eps is the torch default 1e-5.  All four 1-D buffers have shape [C].  x is
// rank-N (typically rank-3 [C, H, W] post unit-extent fold; the channel axis is
// the FIRST non-unit dim, i.e. x.shape[0]).  Host mode: CPU reference; device:
// aclnnBatchNorm with training=false (not yet wired — falls back to CPU).
void run_BatchNorm(
    TensorInfo x,
    TensorInfo weight,
    TensorInfo bias,
    TensorInfo running_mean,
    TensorInfo running_var,
    TensorInfo *out,
    aclrtStream stream);

// CPU-reference 2D pooling (aclnn-fallback for linalg.pooling_nchw_max /
// linalg.pooling_nchw_sum).  Layouts: input [N, C, H, W] -> out [N, C, OH, OW].
// MaxPool: out = max over window; SumPool: out = sum over window (no division —
// avg is the caller's responsibility, ResNet's adaptive_avg_pool divides by a
// constant downstream).  strides / kernel_size / dilations point to int64_t[2].
// Padding is materialized upstream as tensor.pad; the kernel receives the
// already-padded input.
void run_MaxPool2D(
    TensorInfo in,
    const int64_t *kernel_size,    // [KH, KW]
    const int64_t *strides,        // [SH, SW]
    const int64_t *dilations,      // [DH, DW]
    TensorInfo *out,
    aclrtStream stream);

void run_SumPool2D(
    TensorInfo in,
    const int64_t *kernel_size,
    const int64_t *strides,
    const int64_t *dilations,
    TensorInfo *out,
    aclrtStream stream);

// CPU-reference 2D convolution (aclnn-fallback for linalg.conv_2d_nchw_fchw).
// Layouts: input  [N, C, H, W],  weight [F, C, KH, KW] -> out [N, F, OH, OW]
// where OH = (H - DH*(KH-1) - 1) / SH + 1 and OW analogously.
// strides/dilations point to int64_t[2] (= [SH, SW], [DH, DW]).
// Padding is materialized upstream as tensor.pad — the kernel sees the already
// padded input, so this routine does NOT apply any padding itself.
// Host mode: CPU reference; device mode: not yet wired (falls back to CPU).
void run_Conv2D(
    TensorInfo in,
    TensorInfo weight,
    TensorInfo init,               // DPS init/output buffer (unused)
    const int64_t *strides,        // [SH, SW]
    const int64_t *dilations,      // [DH, DW]
    TensorInfo *out,
    aclrtStream stream);

// CPU-reference Transpose (aclnn-fallback; the AscendC transpose codegen is
// unreliable).  out[i] = in[j] where j[perm[d]] = i[d], i.e. out shape =
// permute(in shape, perm).  perm has `rank` entries.  Host mode: CPU reference;
// device: aclnnPermute.
void run_Transpose(
    TensorInfo in,
    const int64_t *perm,
    int rank,
    TensorInfo *out, // allocated + written by this call
    aclrtStream stream);

} // namespace mlir::runtime::aclnn
