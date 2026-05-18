// ============================================================
// STAGE 0: High-Level IR - three generated kernels in one module
//
// Computation:
//   kernel_a: a[N] + b[N] -> mid0[N]
//   kernel_b: mid0[N] * c[N] -> mid1[N]
//   kernel_c: mid1[N] + d[N] -> out[N]
//
// The example validates runtime-session DAG execution across a chain of three
// generated kernels, with each downstream task consuming the previous task's
// output through source: "task_output".
// ============================================================

module {
  func.func @kernel_a(
      %a : tensor<?xf16>,
      %b : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %a, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %out = linalg.generic {
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
    return %out : tensor<?xf16>
  }

  func.func @kernel_b(
      %mid0 : tensor<?xf16>,
      %c : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %mid0, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%mid0, %c : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %prod = arith.mulf %x, %y : f16
      linalg.yield %prod : f16
    } -> tensor<?xf16>
    return %out : tensor<?xf16>
  }

  func.func @kernel_c(
      %mid1 : tensor<?xf16>,
      %d : tensor<?xf16>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %mid1, %c0 : tensor<?xf16>
    %init = tensor.empty(%n) : tensor<?xf16>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%mid1, %d : tensor<?xf16>, tensor<?xf16>)
      outs(%init : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %sum = arith.addf %x, %y : f16
      linalg.yield %sum : f16
    } -> tensor<?xf16>
    return %out : tensor<?xf16>
  }
}
