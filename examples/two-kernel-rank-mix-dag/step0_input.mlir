// ============================================================
// STAGE 0: High-Level IR - rank-mixed two-kernel DAG
//
// Computation:
//   kernel_a: a[N] + b[N] -> mid[N]
//   kernel_b: mid[N] + scale[N, K] -> out[N, K]
//
// This keeps the upstream producer rank-1 and the consumer/output rank-2 so
// runtime-session must preserve the task-output edge across different tensor
// ranks instead of relying on a uniform single-rank example.
// ============================================================

module {
  func.func @kernel_a(
      %a : tensor<?xf16>,
      %b : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %a, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %mid = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %sum = arith.addf %x, %y : f16
      linalg.yield %sum : f16
    } -> tensor<?xf16>
    return %mid : tensor<?xf16>
  }

  func.func @kernel_b(
      %mid : tensor<?xf16>,
      %scale : tensor<?x?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %scale, %c0 : tensor<?x?xf16>
    %k = tensor.dim %scale, %c1 : tensor<?x?xf16>
    %init = tensor.empty(%n, %k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mid, %scale : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init : tensor<?x?xf16>) {
    ^bb0(%x: f16, %s: f16, %o: f16):
      %v = arith.addf %x, %s : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
    return %out : tensor<?x?xf16>
  }
}
