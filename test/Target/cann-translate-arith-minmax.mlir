// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void scalar_minmax(
// CHECK: {{v[0-9]+}} >
// CHECK: ? {{v[0-9]+}} : {{v[0-9]+}}
// CHECK: {{v[0-9]+}} <
// CHECK: ? {{v[0-9]+}} : {{v[0-9]+}}

module {
  func.func @scalar_minmax(
      %input: memref<4xi32>,
      %output: memref<4xi32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = memref.load %input[%c0] : memref<4xi32>
    %b = memref.load %input[%c1] : memref<4xi32>
    %max = arith.maxsi %a, %b : i32
    %min = arith.minsi %a, %b : i32
    %sum = arith.addi %max, %min : i32
    memref.store %sum, %output[%c0] : memref<4xi32>
    func.return
  }
}
