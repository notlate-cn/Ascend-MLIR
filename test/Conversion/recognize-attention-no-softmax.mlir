// RUN: afir-opt --recognize-attention %s | FileCheck %s

// Two batch_matmuls with a bare elementwise exp between them -- but NO softmax
// normalization (no reduction over the key dim).  This is the shape of a
// kernelized / linear-attention block, or simply two GEMMs with an exp
// activation; it is NOT softmax attention.
//
//   bmm1:  Q[2,4,8] @ K^T[2,8,4]  -> scores[2,4,4]
//   act:   exp(scores)            (elementwise, no reduction)  -> probs[2,4,4]
//   bmm2:  probs @ V[2,4,8]       -> out[2,4,8]
//
// recognize-attention must NOT fold this to FlashAttentionScore: doing so
// substitutes a softmax-normalized result for a non-normalized one (wrong).
// The recognizer requires a softmax reduction (the sum over keys), not just an
// exp, between the two bmms.

func.func @bmm_exp_bmm_no_softmax(%q: tensor<2x4x8xf32>, %kt: tensor<2x8x4xf32>,
                                  %v: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %s0 = tensor.empty() : tensor<2x4x4xf32>
  %s1 = linalg.fill ins(%cst : f32) outs(%s0 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %scores = linalg.batch_matmul ins(%q, %kt : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
            outs(%s1 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

  // bare exp activation -- NO reduction / normalization.
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
  %out = linalg.batch_matmul ins(%probs, %v : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
         outs(%o1 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

  return %out : tensor<2x4x8xf32>
}

// No softmax reduction -> must NOT fold.
// CHECK-LABEL: func.func @bmm_exp_bmm_no_softmax
// CHECK: linalg.batch_matmul
// CHECK-NOT: __aclnn_flash_attention
