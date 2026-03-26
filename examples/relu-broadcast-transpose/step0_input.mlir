// ============================================================
// STAGE 0: Named Linalg Op Source IR
//
// Computation graph:
//   data0[m,1] -> relu -> transpose[1,0] -> broadcast dim[0]
//                                              |
//                         data1[n,m] -------> add -> out[n,m]
//
// Named ops match StableHLO/torch-MLIR lowering output.
// Three fusion scenarios covered:
//   Elementwise+Transpose: relu -> transpose
//   Transpose+Broadcast:   transpose -> broadcast
//   Transpose+Elementwise: broadcast -> add (with data1)
//
// RUN: afir-opt --linalg-generalize-named-ops \
// RUN:          --linalg-fuse-elementwise-ops %s \
// RUN:          --canonicalize --cse --mlir-print-local-scope | FileCheck %s
// CHECK: linalg.generic
// CHECK: affine_map<(d0, d1) -> (d1, 0)>
// CHECK: affine_map<(d0, d1) -> (d0, d1)>
// ============================================================

module {
  func.func @relu_transpose_broadcast_add(
      %data0: tensor<?x1xf16>,
      %data1: tensor<?x?xf16>) -> tensor<?x?xf16> {

    %c0 = arith.constant 0 : index
    %m = tensor.dim %data0, %c0 : tensor<?x1xf16>
    %n = tensor.dim %data1, %c0 : tensor<?x?xf16>

    // relu: max(x, 0) over [m,1]
    %zero_f16 = arith.constant 0.0 : f16
    %relu_init = tensor.empty(%m) : tensor<?x1xf16>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%data0 : tensor<?x1xf16>)
      outs(%relu_init : tensor<?x1xf16>) {
    ^bb0(%x: f16, %out: f16):
      %r = arith.maximumf %x, %zero_f16 : f16
      linalg.yield %r : f16
    } -> tensor<?x1xf16>

    // transpose+broadcast+add fused as a single generic:
    //   - relu result [m,1] is read at (d1, 0): affine_map<(d0,d1) -> (d1, 0)>
    //   - data1 [n,m] is read at (d0, d1):      affine_map<(d0,d1) -> (d0, d1)>
    //   - output [n,m] is written at (d0, d1):  affine_map<(d0,d1) -> (d0, d1)>
    // Iterator space: d0 = n, d1 = m
    %out_init = tensor.empty(%n, %m) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, 0)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu, %data1 : tensor<?x1xf16>, tensor<?x?xf16>)
      outs(%out_init : tensor<?x?xf16>) {
    ^bb0(%relu_val: f16, %d1_val: f16, %out: f16):
      %sum = arith.addf %relu_val, %d1_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    return %result : tensor<?x?xf16>
  }
}
