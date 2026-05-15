// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void shape_views(
// CHECK-COUNT-2: reinterpret_cast<float*>
// CHECK: float {{v[0-9]+}} =

module {
  func.func @shape_views(
      %input: memref<?x?xf32>,
      %output: memref<?x?xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0 = arith.constant 0 : index
    %expanded = memref.expand_shape %input [[0, 1], [2]]
        output_shape [1, %c0, %c0] : memref<?x?xf32> into memref<1x?x?xf32>
    %collapsed = memref.collapse_shape %expanded [[0, 1], [2]]
        : memref<1x?x?xf32> into memref<?x?xf32>
    %v = memref.load %collapsed[%c0, %c0] : memref<?x?xf32>
    memref.store %v, %output[%c0, %c0] : memref<?x?xf32>
    func.return
  }
}
