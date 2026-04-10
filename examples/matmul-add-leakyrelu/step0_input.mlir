// examples/matmul-add-leakyrelu/step0_input.mlir
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @matmul_add_leakyrelu

// bias[N] broadcast 映射：(d0,d1) -> (d1)  ← 沿 M 轴广播
#bias_map    = affine_map<(d0, d1) -> (d1)>
#full_map    = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @matmul_add_leakyrelu(
      %a    : tensor<?x?xf16>,   // [M, K]
      %b    : tensor<?x?xf16>,   // [K, N]
      %bias : tensor<?xf32>,     // [N]
      %out  : tensor<?x?xf32>    // [M, N] (init all-zero)
  ) -> tensor<?x?xf32> {

    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %dim_m = tensor.dim %out, %idx0 : tensor<?x?xf32>
    %dim_n = tensor.dim %out, %idx1 : tensor<?x?xf32>

    // ── Op 1: matmul  A[M,K] × B[K,N] → C[M,N] (f32) ──────────────
    %c = linalg.matmul
      ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out  : tensor<?x?xf32>)
      -> tensor<?x?xf32>

    // ── Op 2: add bias (broadcast [N] → [M,N]) ──────────────────────
    %empty_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %d = linalg.generic {
      indexing_maps = [#full_map, #bias_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%c, %bias : tensor<?x?xf32>, tensor<?xf32>)
      outs(%empty_d : tensor<?x?xf32>) {
    ^bb0(%c_val: f32, %bias_val: f32, %out_val: f32):
      %sum = arith.addf %c_val, %bias_val : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>

    // ── Op 3: leaky_relu(x, 0.001) = max(x, x*0.001) ────────────────
    %alpha = arith.constant 1.0e-3 : f32
    %empty_e = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %e = linalg.generic {
      indexing_maps = [#full_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%d : tensor<?x?xf32>)
      outs(%empty_e : tensor<?x?xf32>) {
    ^bb0(%x: f32, %out_val: f32):
      %scaled = arith.mulf %x, %alpha : f32
      %result = arith.maximumf %x, %scaled : f32
      linalg.yield %result : f32
    } -> tensor<?x?xf32>

    return %e : tensor<?x?xf32>
  }
}
