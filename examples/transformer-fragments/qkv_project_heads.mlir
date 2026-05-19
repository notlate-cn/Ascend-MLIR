// ============================================================
// Transformer fragment: QKV projection to heads.
//
// Computes the first attention-front slice in one kernel:
//   qkv[tokens, 48] = x[tokens, 16] @ weight[16, 48] + bias[48]
//   out[tokens, 3, 2, 8] = reshape(qkv)
// This combines the matmul-bearing QKV projection with the next head-split
// view step while staying below batch_matmul / attention score complexity.
// ============================================================

#bias_map = affine_map<(d0, d1) -> (d1)>
#full = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @kernel(
      %x : tensor<?x?xf16>,
      %weight : tensor<?x48xf16>,
      %bias : tensor<48xf32>,
      %init : tensor<?x48xf32>) -> tensor<?x3x2x8xf32> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %init, %c0 : tensor<?x48xf32>

    %projected = linalg.matmul
      ins(%x, %weight : tensor<?x?xf16>, tensor<?x48xf16>)
      outs(%init : tensor<?x48xf32>) -> tensor<?x48xf32>

    %biased_empty = tensor.empty(%m) : tensor<?x48xf32>
    %biased = linalg.generic {
      indexing_maps = [#full, #bias_map, #full],
      iterator_types = ["parallel", "parallel"]
    } ins(%projected, %bias : tensor<?x48xf32>, tensor<48xf32>)
      outs(%biased_empty : tensor<?x48xf32>) {
    ^bb0(%v: f32, %b: f32, %old: f32):
      %r = arith.addf %v, %b : f32
      linalg.yield %r : f32
    } -> tensor<?x48xf32>

    %flat_empty = tensor.empty(%m) : tensor<?x48xf32>
    %flat = linalg.generic {
      indexing_maps = [#full, #full],
      iterator_types = ["parallel", "parallel"]
    } ins(%biased : tensor<?x48xf32>)
      outs(%flat_empty : tensor<?x48xf32>) {
    ^bb0(%v: f32, %old: f32):
      linalg.yield %v : f32
    } -> tensor<?x48xf32>

    %heads = tensor.expand_shape %flat [[0], [1, 2, 3]]
        output_shape [%m, 3, 2, 8]
        : tensor<?x48xf32> into tensor<?x3x2x8xf32>

    return %heads : tensor<?x3x2x8xf32>
  }
}
