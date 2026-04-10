// ============================================================
// STAGE 0: High-Level IR - torch-MLIR style (no library_call)
//
// Computation graph: relu -> index_select(dim=1) -> add
//
//   data[M, N]   --relu-->  relu_out[M, N]
//   indices[K]   --------> index_select(dim=1) -->  gathered[M, K]
//   bias[K]      ----------------------------------> add --> out[M, K]
//
// Op2 (index_select, dim=1): linalg.generic, body: tensor.extract
//   Semantics: out[i, j] = relu_out[i, indices[j]]
//   Matches torch-MLIR aten.index_select(data, dim=1, indices) lowering.
//   --mark-structured-ops will stamp {gather_dim = 1 : i64} on this op.
//
// M, N, K are all dynamic dimensions.
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @relu_index_select_add

#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
#full_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @relu_index_select_add(
      %data    : tensor<?x?xf16>,
      %indices : tensor<?xi64>,
      %bias    : tensor<?xf16>
  ) -> tensor<?x?xf16> {

    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16

    %dim_m = tensor.dim %data,    %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data,    %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>

    // Op1: relu(data[M,N]) -> relu_out[M,N]
    %empty_relu = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %relu_out = linalg.generic {
      indexing_maps = [#full_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%data : tensor<?x?xf16>)
      outs(%empty_relu : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // Op2: index_select(relu_out, dim=1, indices) -> gathered[M,K]
    // --mark-structured-ops stamps {gather_dim = 1 : i64} on this op.
    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col_broadcast_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi64>)
      outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %out: f16):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %idx_cast = arith.index_cast %idx : i64 to index
      %val = tensor.extract %relu_out[%i, %idx_cast] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?xf16>

    // Op3: gathered[M,K] + bias[K] -> out[M,K]
    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#full_map, #col_broadcast_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%gathered, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%g: f16, %b: f16, %o: f16):
      %v = arith.addf %g, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }
}
