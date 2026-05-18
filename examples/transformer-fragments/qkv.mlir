// ============================================================
// Transformer fragment: QKV projection.
//
// This is the first matmul-bearing fragment in the ladder. It keeps the token
// dimension flattened:
//   out[B*S, 3*H] = x[B*S, H] @ weight[H, 3*H] + bias[3*H]
// The final max(x, x*1.0) is a no-op that keeps the IR inside the currently
// supported mix epilogue shell. Default data uses H=16 and 3*H=48 before
// scaling back to 128 -> 384 and rank3 batch_matmul.
// ============================================================

#bias_map = affine_map<(d0, d1) -> (d1)>
#full = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @kernel(
      %x : tensor<?x?xf16>,
      %weight : tensor<?x?xf16>,
      %bias : tensor<?xf32>,
      %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %init, %c0 : tensor<?x?xf32>
    %n = tensor.dim %init, %c1 : tensor<?x?xf32>

    %projected = linalg.matmul
      ins(%x, %weight : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>

    %biased_empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %biased = linalg.generic {
      indexing_maps = [#full, #bias_map, #full],
      iterator_types = ["parallel", "parallel"]
    } ins(%projected, %bias : tensor<?x?xf32>, tensor<?xf32>)
      outs(%biased_empty : tensor<?x?xf32>) {
    ^bb0(%v: f32, %b: f32, %old: f32):
      %r = arith.addf %v, %b : f32
      linalg.yield %r : f32
    } -> tensor<?x?xf32>

    %one = arith.constant 1.000000e+00 : f32
    %out_empty = tensor.empty(%m, %n) : tensor<?x?xf32>
    %out = linalg.generic {
      indexing_maps = [#full, #full],
      iterator_types = ["parallel", "parallel"]
    } ins(%biased : tensor<?x?xf32>)
      outs(%out_empty : tensor<?x?xf32>) {
    ^bb0(%v: f32, %old: f32):
      %same = arith.mulf %v, %one : f32
      %r = arith.maximumf %v, %same : f32
      linalg.yield %r : f32
    } -> tensor<?x?xf32>

    return %out : tensor<?x?xf32>
  }
}
