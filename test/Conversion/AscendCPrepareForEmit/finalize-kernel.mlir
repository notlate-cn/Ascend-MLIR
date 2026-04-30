// RUN: afir-opt %s --ascendc-finalize-kernel 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @my_kernel(
// CHECK-SAME: ascendc.aicore
// CHECK-SAME: ascendc.global
// CHECK-NOT: return %
// CHECK: arith.minsi

func.func @my_kernel(%a: memref<f32>) -> memref<f32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %min = affine.min affine_map<(d0)[s0] -> (d0, s0)>(%c0)[%c8]
  return %a : memref<f32>
}
