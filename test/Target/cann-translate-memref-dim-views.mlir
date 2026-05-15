// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void dim_views(
// CHECK: static_cast<uint32_t>
// CHECK-NOT: memref.dim

module {
  func.func @dim_views(
      %input: memref<?x?xf32>,
      %shape: memref<2xi64>,
      %output: memref<?x?xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32} {
    %m = emitasc.member %tiling "M"
        : !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>, i64
    %n = emitasc.member %tiling "N"
        : !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>, i64
    %mi = arith.index_cast %m : i64 to index
    %ni = arith.index_cast %n : i64 to index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %m, %shape[%c0] : memref<2xi64>
    memref.store %n, %shape[%c1] : memref<2xi64>
    %reshape = memref.reshape %input(%shape)
        : (memref<?x?xf32>, memref<2xi64>) -> memref<?x?xf32>
    %d0 = memref.dim %reshape, %c0 : memref<?x?xf32>
    %expanded = memref.expand_shape %reshape [[0, 1], [2]]
        output_shape [1, %mi, %ni] : memref<?x?xf32> into memref<1x?x?xf32>
    %collapsed = memref.collapse_shape %expanded [[0, 1], [2]]
        : memref<1x?x?xf32> into memref<?x?xf32>
    %d1 = memref.dim %collapsed, %c1 : memref<?x?xf32>
    scf.for %i = %c0 to %d0 step %c1 {
      scf.for %j = %c0 to %d1 step %c1 {
        %v = memref.load %reshape[%i, %j] : memref<?x?xf32>
        memref.store %v, %output[%i, %j] : memref<?x?xf32>
      }
    }
    func.return
  }
}
