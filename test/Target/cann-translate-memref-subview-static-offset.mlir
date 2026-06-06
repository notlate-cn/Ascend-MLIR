// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void static_offset_subview(
// CHECK: * {{v[0-9]+}};
// CHECK: {{v[0-9]+}} = {{v[0-9]+}} * {{v[0-9]+}};
// CHECK-NOT: float* {{v[0-9]+}} =
// CHECK: float {{v[0-9]+}} = ascend_gm_load<float>(
// CHECK: ascend_gm_store<float>(

module {
  func.func @static_offset_subview(
      %input: memref<3x?x?x128xf32>,
      %output: memref<1x?x?x128xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %m = emitasc.member %tiling "M"
        : !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>, i64
    %n = emitasc.member %tiling "N"
        : !emitasc.py_struct<"TilingData", [i64, i64], ["M", "N"]>, i64
    %mi = arith.index_cast %m : i64 to index
    %ni = arith.index_cast %n : i64 to index
    %c0 = arith.constant 0 : index
    %sub = memref.subview %input[1, 0, 0, 0] [1, %mi, %ni, 128] [1, 1, 1, 1]
        : memref<3x?x?x128xf32> to memref<1x?x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>
    %v = memref.load %sub[%c0, %c0, %c0, %c0]
        : memref<1x?x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>
    memref.store %v, %output[%c0, %c0, %c0, %c0] : memref<1x?x?x128xf32>
    func.return
  }
}
