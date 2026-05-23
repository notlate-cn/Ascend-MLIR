// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

module {
  func.func @co1_vecin_scalar_loop(%out: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %co1 = memref.alloc() : memref<4xf32, 7 : i32>
    %vecin = memref.alloc() : memref<4xf32, 9 : i32>
    memref.copy %co1, %vecin : memref<4xf32, 7 : i32> to memref<4xf32, 9 : i32>
    scf.for %i = %c0 to %c4 step %c1 {
      %v = memref.load %vecin[%i] : memref<4xf32, 9 : i32>
      memref.store %v, %out[%i] : memref<4xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @co1_vecin_scalar_loop(
// CHECK: ascendc.data_copy_co12dst
// CHECK: ascendc.local_tensor.get_value
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.load
