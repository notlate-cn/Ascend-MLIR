// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#broadcast_transpose = affine_map<(d0, d1) -> (d1, 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @selected_all_parallel_tile_materializes_broadcast_transpose
// CHECK: scf.for %{{.*}} = %c0 to %c70 step %c64
// CHECK: ascendc.broadcast_l2
// CHECK: ascendc.add_l2
// CHECK: memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}, %c128] [1, 1] : memref<70x128xf16>
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: linalg.generic
func.func @selected_all_parallel_tile_materializes_broadcast_transpose() {
  %a = memref.alloc() : memref<128x1xf16, 9 : i32>
  %b = memref.alloc() : memref<70x128xf16, 9 : i32>
  %out = memref.alloc() : memref<70x128xf16, 10 : i32>
  %gm = memref.alloc() : memref<70x128xf16>
  linalg.generic {
      indexing_maps = [#broadcast_transpose, #identity, #identity],
      iterator_types = ["parallel", "parallel"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>}
      ins(%a, %b : memref<128x1xf16, 9 : i32>,
                    memref<70x128xf16, 9 : i32>)
      outs(%out : memref<70x128xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %b_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %b_elem : f16
      linalg.yield %sum : f16
  }
  memref.copy %out, %gm : memref<70x128xf16, 10 : i32> to memref<70x128xf16>
  return
}

// CHECK-LABEL: func.func @selected_all_parallel_tile_fallback_for_intervening_out_use
// CHECK-NOT: scf.for
// CHECK: ascendc.add_l2
// CHECK: memref.store
// CHECK-NOT: linalg.generic
func.func @selected_all_parallel_tile_fallback_for_intervening_out_use() {
  %a = memref.alloc() : memref<70x128xf16, 9 : i32>
  %b = memref.alloc() : memref<70x128xf16, 9 : i32>
  %out = memref.alloc() : memref<70x128xf16, 10 : i32>
  %gm = memref.alloc() : memref<70x128xf16>
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>}
      ins(%a, %b : memref<70x128xf16, 9 : i32>,
                    memref<70x128xf16, 9 : i32>)
      outs(%out : memref<70x128xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %b_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %b_elem : f16
      linalg.yield %sum : f16
  }
  memref.store %zero, %out[%c0, %c0] : memref<70x128xf16, 10 : i32>
  memref.copy %out, %gm : memref<70x128xf16, 10 : i32> to memref<70x128xf16>
  return
}

// CHECK-LABEL: func.func @selected_all_parallel_tile_fallback_for_intervening_input_write
// CHECK-NOT: scf.for
// CHECK: ascendc.add_l2
// CHECK: memref.store
// CHECK-NOT: linalg.generic
func.func @selected_all_parallel_tile_fallback_for_intervening_input_write() {
  %a = memref.alloc() : memref<70x128xf16, 9 : i32>
  %b = memref.alloc() : memref<70x128xf16, 9 : i32>
  %out = memref.alloc() : memref<70x128xf16, 10 : i32>
  %gm = memref.alloc() : memref<70x128xf16>
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f16
  linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>}
      ins(%a, %b : memref<70x128xf16, 9 : i32>,
                    memref<70x128xf16, 9 : i32>)
      outs(%out : memref<70x128xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %b_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %b_elem : f16
      linalg.yield %sum : f16
  }
  memref.store %zero, %a[%c0, %c0] : memref<70x128xf16, 9 : i32>
  memref.copy %out, %gm : memref<70x128xf16, 10 : i32> to memref<70x128xf16>
  return
}
