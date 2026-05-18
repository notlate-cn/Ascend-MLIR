// ============================================================
// STAGE 0: High-Level IR - RMSNorm reduction core DAG
//
// Computation:
//   kernel_square: x[m, n] * x[m, n] -> square[m, n]
//   kernel_reduce: sum_n(square[m, n]) -> sumsq[m]
//   kernel_scale:  x[m, n] * sumsq[m] -> out[m, n]
//
// This intentionally excludes rsqrt/epsilon so the demo exercises the current
// commercial-critical reduction + broadcast dataflow without relying on math
// ops that are not yet lowered by the AscendC backend.
// ============================================================

#row = affine_map<(d0, d1) -> (d0)>
#full = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @kernel_square(%x : tensor<?x?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %x, %c0 : tensor<?x?xf16>
    %n = tensor.dim %x, %c1 : tensor<?x?xf16>
    %init = tensor.empty(%m, %n) : tensor<?x?xf16>
    %square = linalg.generic {
      indexing_maps = [#full, #full],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<?x?xf16>)
      outs(%init : tensor<?x?xf16>) {
    ^bb0(%v: f16, %o: f16):
      %sq = arith.mulf %v, %v : f16
      linalg.yield %sq : f16
    } -> tensor<?x?xf16>
    return %square : tensor<?x?xf16>
  }

  func.func @kernel_reduce(%square : tensor<?x?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %square, %c0 : tensor<?x?xf16>
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty(%m) : tensor<?xf16>
    %init = linalg.fill ins(%zero : f16)
      outs(%empty : tensor<?xf16>) -> tensor<?xf16>
    %sumsq = linalg.generic {
      indexing_maps = [#full, #row],
      iterator_types = ["parallel", "reduction"]
    } ins(%square : tensor<?x?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%sq: f16, %acc: f16):
      %next = arith.addf %acc, %sq : f16
      linalg.yield %next : f16
    } -> tensor<?xf16>
    return %sumsq : tensor<?xf16>
  }

  func.func @kernel_scale(
      %x : tensor<?x?xf16>,
      %sumsq : tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %x, %c0 : tensor<?x?xf16>
    %n = tensor.dim %x, %c1 : tensor<?x?xf16>
    %init = tensor.empty(%m, %n) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#full, #row, #full],
      iterator_types = ["parallel", "parallel"]
    } ins(%x, %sumsq : tensor<?x?xf16>, tensor<?xf16>)
      outs(%init : tensor<?x?xf16>) {
    ^bb0(%v: f16, %scale: f16, %o: f16):
      %r = arith.mulf %v, %scale : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>
    return %out : tensor<?x?xf16>
  }
}
