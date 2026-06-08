// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s --implicit-check-not=arith.andi --implicit-check-not=arith.ori --implicit-check-not=arith.xori --implicit-check-not=linalg.generic

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @pyasc_bitwise_binary_generic
// CHECK: ascendc.and_l2
// CHECK: ascendc.or_l2
// CHECK: ascendc.xor
// CHECK: return
func.func @pyasc_bitwise_binary_generic(%lhs: memref<4x8xi32>,
                                        %rhs: memref<4x8xi32>,
                                        %mask: memref<4x8xi32>,
                                        %out: memref<4x8xi32>) {
  linalg.generic {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs, %mask : memref<4x8xi32>, memref<4x8xi32>,
                             memref<4x8xi32>)
      outs(%out : memref<4x8xi32>) {
    ^bb0(%a: i32, %b: i32, %m: i32, %out0: i32):
      %and = arith.andi %a, %b : i32
      %or = arith.ori %and, %m : i32
      %xor = arith.xori %or, %b : i32
      linalg.yield %xor : i32
    }
  return
}
