// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void reshape_view(
// CHECK-NOT: reinterpret_cast<float*>
// CHECK: float {{v[0-9]+}} = afir_gm_load<float>(
// CHECK: afir_gm_store<float>(

module {
  func.func @reshape_view(
      %input: memref<?x?xf32>,
      %shape: memref<3xi64>,
      %output: memref<?x?x1xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32} {
    %c0 = arith.constant 0 : index
    %view = memref.reshape %input(%shape)
        : (memref<?x?xf32>, memref<3xi64>) -> memref<?x?x1xf32>
    %v = memref.load %view[%c0, %c0, %c0] : memref<?x?x1xf32>
    memref.store %v, %output[%c0, %c0, %c0] : memref<?x?x1xf32>
    func.return
  }
}
