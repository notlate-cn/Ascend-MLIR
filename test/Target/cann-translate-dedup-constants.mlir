// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: void dedup_constants
// CHECK: constexpr uint32_t c0_idx = 0;
// CHECK-NOT: constexpr uint32_t c0_idx = 0;
// CHECK: return;

module {
  func.func @dedup_constants(
      %arg0: memref<?xf16>,
      %arg1: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %v = memref.load %arg0[%c0] : memref<?xf16>
    memref.store %v, %arg1[%c0_0] : memref<?xf16>
    return
  }
}
