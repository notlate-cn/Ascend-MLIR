// RUN: ascend-mlir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void guarded_kernel(
// CHECK: bool {{v[0-9]+}} =
// CHECK: if (!{{v[0-9]+}}) {
// CHECK-NEXT: return;
// CHECK-NEXT: }

module {
  func.func @guarded_kernel(
      %input: memref<4xf32>,
      %output: memref<4xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ok = arith.cmpi sle, %c0, %c1 : index
    cf.assert %ok, "shape guard failed"
    func.return
  }
}
