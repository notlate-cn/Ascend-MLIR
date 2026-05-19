// ============================================================
// Transformer fragment: QKV reshape to heads.
//
// Consumes a flattened QKV projection result:
//   qkv[tokens, 48] = [Q hidden=16 | K hidden=16 | V hidden=16]
// and exposes the split-head view:
//   out[tokens, 3, 2, 8] = [Q/K/V, heads=2, head_dim=8]
// This keeps the ladder step focused on reshape/head-split movement without
// adding another matmul or relying on offset-indexed layout transforms.
// ============================================================

#rank4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @kernel(%qkv : tensor<?x48xf32>) -> tensor<?x3x2x8xf32> {
    %c0 = arith.constant 0 : index
    %tokens = tensor.dim %qkv, %c0 : tensor<?x48xf32>

    %heads = tensor.expand_shape %qkv [[0], [1, 2, 3]]
        output_shape [%tokens, 3, 2, 8]
        : tensor<?x48xf32> into tensor<?x3x2x8xf32>

    %out_empty = tensor.empty(%tokens) : tensor<?x3x2x8xf32>
    %out = linalg.generic {
      indexing_maps = [#rank4, #rank4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%heads : tensor<?x3x2x8xf32>)
      outs(%out_empty : tensor<?x3x2x8xf32>) {
    ^bb0(%v: f32, %old: f32):
      linalg.yield %v : f32
    } -> tensor<?x3x2x8xf32>

    return %out : tensor<?x3x2x8xf32>
  }
}
