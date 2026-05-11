// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK-LABEL: func.func @copy_gm_to_vecin
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: memref.copy
func.func @copy_gm_to_vecin(%src: memref<?x?xf32>) {
  %dst = memref.alloc() : memref<16x16xf32, 9 : i32>
  memref.copy %src, %dst : memref<?x?xf32> to memref<16x16xf32, 9 : i32>
  return
}

// CHECK-LABEL: func.func @vector_add
// CHECK: ascendc.add_l2
// CHECK-NOT: linalg.elementwise
func.func @vector_add() {
  %src0 = memref.alloc() : memref<32x32xf32, 9 : i32>
  %src1 = memref.alloc() : memref<32x32xf32, 9 : i32>
  %dst = memref.alloc() : memref<32x32xf32, 11 : i32>
  linalg.elementwise kind=#linalg.elementwise_kind<add>
      {ascendc.unit = "AiCore.Vector"}
      ins(%src0, %src1 : memref<32x32xf32, 9 : i32>,
                       memref<32x32xf32, 9 : i32>)
      outs(%dst : memref<32x32xf32, 11 : i32>)
  return
}
