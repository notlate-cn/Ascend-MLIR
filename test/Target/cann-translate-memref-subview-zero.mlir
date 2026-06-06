// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void zero_subview(
// CHECK-NOT: reinterpret_cast<float*>
// CHECK: float {{v[0-9]+}} = ascend_gm_load<float>(
// CHECK: ascend_gm_store<float>(

module {
  func.func @zero_subview(
      %input: memref<?x?xf32>,
      %output: memref<?x?xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0 = arith.constant 0 : index
    %sub = memref.subview %input[0, 0] [%c0, %c0] [1, 1]
        : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
    %v = memref.load %sub[%c0, %c0] : memref<?x?xf32, strided<[?, 1]>>
    memref.store %v, %output[%c0, %c0] : memref<?x?xf32>
    func.return
  }
}
