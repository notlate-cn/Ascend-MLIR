// RUN: afir-opt --recognize-attention %s | FileCheck %s

// An attention region whose mask add carries a mask shape the
// FlashAttentionScore [1,1,S,S] contract cannot represent (here a rank-3
// per-batch [BH,S,S] mask, not the supported rank-2 [S,S]).
//
//   bmm1:  Q[2,4,8] @ K^T[2,8,4]   -> scores[2,4,4]
//   maskadd: scores + mask[2,4,4]  (rank-3, unsupported)
//   exp:   softmax stand-in        -> probs[2,4,4]
//   bmm2:  probs @ V[2,4,8]        -> out[2,4,8]
//
// recognize-attention must NOT fold this to a FlashAttentionScore call, because
// doing so would silently drop the mask (-> bidirectional attention, wrong
// result).  It must leave the explicit mask add + batch_matmuls in place so the
// generic path lowers them faithfully.

func.func @attn_unsupported_mask(%q: tensor<2x4x8xf32>, %kt: tensor<2x8x4xf32>,
                                 %v: tensor<2x4x8xf32>,
                                 %mask: tensor<2x4x4xf32>) -> tensor<2x4x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %s0 = tensor.empty() : tensor<2x4x4xf32>
  %s1 = linalg.fill ins(%cst : f32) outs(%s0 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %scores = linalg.batch_matmul ins(%q, %kt : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
            outs(%s1 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

  // rank-3 [BH,S,S] mask add (per-batch mask) — unsupported by the [1,1,S,S] contract.
  %m0 = tensor.empty() : tensor<2x4x4xf32>
  %masked = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                                             affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                                             affine_map<(d0,d1,d2)->(d0,d1,d2)>],
                            iterator_types = ["parallel","parallel","parallel"]}
            ins(%scores, %mask : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
            outs(%m0 : tensor<2x4x4xf32>) {
  ^bb0(%sc: f32, %mk: f32, %o: f32):
    %a = arith.addf %sc, %mk : f32
    linalg.yield %a : f32
  } -> tensor<2x4x4xf32>

  // softmax stand-in: exp (recognition keys on exp between the two bmms)
  %p0 = tensor.empty() : tensor<2x4x4xf32>
  %probs = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                                            affine_map<(d0,d1,d2)->(d0,d1,d2)>],
                           iterator_types = ["parallel","parallel","parallel"]}
           ins(%masked : tensor<2x4x4xf32>) outs(%p0 : tensor<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<2x4x4xf32>

  %o0 = tensor.empty() : tensor<2x4x8xf32>
  %o1 = linalg.fill ins(%cst : f32) outs(%o0 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  %out = linalg.batch_matmul ins(%probs, %v : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
         outs(%o1 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

  return %out : tensor<2x4x8xf32>
}

// The unsupported mask must keep the region un-folded: no FlashAttentionScore
// (neither decl nor call), and the batch_matmuls survive.
// CHECK-LABEL: func.func @attn_unsupported_mask
// CHECK: linalg.batch_matmul
// CHECK-NOT: __aclnn_flash_attention
