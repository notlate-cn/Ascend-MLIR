// ============================================================
// Transformer fragment: layernorm-like no-matmul slice.
//
// Computes:
//   mean = reduce_sum(x, axis=1) / 128
//   var  = reduce_sum((x - mean)^2, axis=1) / 128
//   out  = (x - mean) * rsqrt(var + 1e-5) * gamma + beta
// ============================================================

#full = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
#vec = affine_map<(d0) -> (d0)>

module {
  func.func @kernel(
      %x : tensor<?x128xf32>,
      %gamma : tensor<128xf32>,
      %beta : tensor<128xf32>) -> tensor<?x128xf32> {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f32
    %hidden = arith.constant 1.280000e+02 : f32
    %eps = arith.constant 1.000000e-05 : f32
    %m = tensor.dim %x, %c0 : tensor<?x128xf32>

    %mean_empty = tensor.empty(%m) : tensor<?xf32>
    %mean_init = linalg.fill ins(%zero : f32)
      outs(%mean_empty : tensor<?xf32>) -> tensor<?xf32>
    %sum = linalg.generic {
      indexing_maps = [#full, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%x : tensor<?x128xf32>)
      outs(%mean_init : tensor<?xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %next = arith.addf %acc, %v : f32
      linalg.yield %next : f32
    } -> tensor<?xf32>

    %mean = linalg.generic {
      indexing_maps = [#vec, #vec],
      iterator_types = ["parallel"]
    } ins(%sum : tensor<?xf32>)
      outs(%mean_empty : tensor<?xf32>) {
    ^bb0(%v: f32, %out: f32):
      %r = arith.divf %v, %hidden : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>

    %var_empty = tensor.empty(%m) : tensor<?xf32>
    %var_init = linalg.fill ins(%zero : f32)
      outs(%var_empty : tensor<?xf32>) -> tensor<?xf32>
    %sq_sum = linalg.generic {
      indexing_maps = [#full, #row, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%x, %mean : tensor<?x128xf32>, tensor<?xf32>)
      outs(%var_init : tensor<?xf32>) {
    ^bb0(%v: f32, %avg: f32, %acc: f32):
      %centered = arith.subf %v, %avg : f32
      %sq = arith.mulf %centered, %centered : f32
      %next = arith.addf %acc, %sq : f32
      linalg.yield %next : f32
    } -> tensor<?xf32>

    %var = linalg.generic {
      indexing_maps = [#vec, #vec],
      iterator_types = ["parallel"]
    } ins(%sq_sum : tensor<?xf32>)
      outs(%var_empty : tensor<?xf32>) {
    ^bb0(%v: f32, %out: f32):
      %r = arith.divf %v, %hidden : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>

    %out_empty = tensor.empty(%m) : tensor<?x128xf32>
    %out = linalg.generic {
      indexing_maps = [#full, #row, #row, #col, #col, #full],
      iterator_types = ["parallel", "parallel"]
    } ins(%x, %mean, %var, %gamma, %beta
        : tensor<?x128xf32>, tensor<?xf32>, tensor<?xf32>, tensor<128xf32>, tensor<128xf32>)
      outs(%out_empty : tensor<?x128xf32>) {
    ^bb0(%v: f32, %avg: f32, %variance: f32, %scale: f32, %bias: f32, %old: f32):
      %centered = arith.subf %v, %avg : f32
      %shifted = arith.addf %variance, %eps : f32
      %inv = math.rsqrt %shifted : f32
      %normalized = arith.mulf %centered, %inv : f32
      %scaled = arith.mulf %normalized, %scale : f32
      %result = arith.addf %scaled, %bias : f32
      linalg.yield %result : f32
    } -> tensor<?x128xf32>
    return %out : tensor<?x128xf32>
  }
}
