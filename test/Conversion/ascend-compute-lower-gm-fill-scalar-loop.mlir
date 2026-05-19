// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @fill_gm_arg(
// CHECK: ascendc.duplicate_l2
// CHECK: scf.for
// CHECK: ascendc.global_tensor.set_global_buffer
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: memref.store
// CHECK-NOT: linalg.fill
func.func @fill_gm_arg(%out: memref<?x4x?xf32>) {
  %cst = arith.constant -3.402823e+38 : f32
  linalg.fill ins(%cst : f32) outs(%out : memref<?x4x?xf32>)
  return
}

// CHECK-LABEL: func.func @fill_gm_alloc(
// CHECK: memref.alloc
// CHECK: ascendc.duplicate_l2
// CHECK: scf.for
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: memref.store
// CHECK-NOT: linalg.fill
func.func @fill_gm_alloc(%d0: index, %d2: index) {
  %alloc = memref.alloc(%d0, %d2) : memref<?x4x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill ins(%cst : f32) outs(%alloc : memref<?x4x?xf32>)
  return
}

// CHECK-LABEL: func.func @fill_gm_i64_falls_back_to_scalar(
// CHECK: scf.for
// CHECK: memref.store {{.*}}, %arg0
// CHECK-NOT: ascendc.duplicate_l2
// CHECK-NOT: linalg.fill
func.func @fill_gm_i64_falls_back_to_scalar(%out: memref<4xi64>) {
  %c0 = arith.constant 0 : i64
  linalg.fill ins(%c0 : i64) outs(%out : memref<4xi64>)
  return
}
