// ============================================================
// Transformer fragment: attention score.
//
// Computes the first rank3 attention score slice plus a vector epilogue:
//   score[batch, query_tokens, key_tokens] =
//       q[batch, query_tokens, k] @ key[batch, k, key_tokens]
//   out = score + bias
//
// The bias epilogue is a materialization anchor for the current mainline
// cube->vector->GM bridge, so this fragment exercises on-chip batch_matmul
// lowering plus the vector consumer/writeback path.
// ============================================================

#full = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @kernel(
      %q : tensor<?x?x?xf16>,
      %key : tensor<?x?x?xf16>,
      %bias : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %batch = tensor.dim %q, %c0 : tensor<?x?x?xf16>
    %query_tokens = tensor.dim %q, %c1 : tensor<?x?x?xf16>
    %key_tokens = tensor.dim %key, %c2 : tensor<?x?x?xf16>

    %score_empty = tensor.empty(%batch, %query_tokens, %key_tokens)
        : tensor<?x?x?xf32>
    %score = linalg.batch_matmul
      ins(%q, %key : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
      outs(%score_empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

    %out_empty = tensor.empty(%batch, %query_tokens, %key_tokens)
        : tensor<?x?x?xf32>
    %out = linalg.generic {
      indexing_maps = [#full, #full, #full],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%score, %bias : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%out_empty : tensor<?x?x?xf32>) {
    ^bb0(%score_elem: f32, %bias_elem: f32, %old: f32):
      %sum = arith.addf %score_elem, %bias_elem : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?xf32>

    return %out : tensor<?x?x?xf32>
  }
}
