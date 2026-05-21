// RUN: afir-opt --recognize-attention %s | FileCheck %s

// Minimal decomposed-attention subgraph (the form a transformer encoder lowers
// to at the linalg level): two linalg.batch_matmul with a softmax (exp) chain
// between them.
//
//   bmm1:  Q[BH,S,D] @ K^T[BH,D,S]  -> scores[BH,S,S]
//   exp-chain (stand-in softmax)    -> probs[BH,S,S]
//   bmm2:  probs[BH,S,S] @ V[BH,S,D] -> out[BH,S,D]
//
// recognize-attention must fold the whole region into a single
// @__aclnn_flash_attention call on BNSD [BH,1,S,D] tensors:
//   Q  = bmm1.lhs           K^T = bmm1.rhs (transpose last 2 -> K)   V = bmm2.rhs
// and replace the bmm2 result, leaving the matmuls/softmax dead for DCE.

func.func @attn(%q: tensor<2x4x8xf32>, %kt: tensor<2x8x4xf32>,
                %v: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %s0 = tensor.empty() : tensor<2x4x4xf32>
  %s1 = linalg.fill ins(%cst : f32) outs(%s0 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  // bmm1: QK^T
  %scores = linalg.batch_matmul ins(%q, %kt : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
            outs(%s1 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

  // softmax stand-in: just exp (recognition keys on exp between the two bmms)
  %p0 = tensor.empty() : tensor<2x4x4xf32>
  %probs = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                                            affine_map<(d0,d1,d2)->(d0,d1,d2)>],
                           iterator_types = ["parallel","parallel","parallel"]}
           ins(%scores : tensor<2x4x4xf32>) outs(%p0 : tensor<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<2x4x4xf32>

  %o0 = tensor.empty() : tensor<2x4x8xf32>
  %o1 = linalg.fill ins(%cst : f32) outs(%o0 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  // bmm2: probs @ V
  %out = linalg.batch_matmul ins(%probs, %v : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
         outs(%o1 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

  return %out : tensor<2x4x8xf32>
}

// The aclnn decl is created, tagged flash_attention, with 5 BNSD inputs.
// CHECK: func.func private @__aclnn_flash_attention
// CHECK: aclnn.kind = "flash_attention"

// CHECK-LABEL: func.func @attn
// K is rebuilt from K^T by transposing the last two dims.
// CHECK: linalg.transpose ins(%arg1 : tensor<2x8x4xf32>){{.*}}permutation = [0, 2, 1]
// Q/K/V are expanded to BNSD [2,1,4,8].
// CHECK: tensor.expand_shape
// CHECK: call @__aclnn_flash_attention
// Result is collapsed back to [2,4,8] and returned; the bmms are gone.
// CHECK: tensor.collapse_shape
// CHECK-NOT: linalg.batch_matmul
