// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#broadcast_transpose = affine_map<(d0, d1) -> (d1, 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @selected_all_parallel_tile_fallback_for_broadcast_transpose
// CHECK-NOT: scf.for
// CHECK: ascendc.broadcast_l2
// CHECK-NOT: scf.for
// CHECK: ascendc.add_l2
// CHECK-NOT: linalg.generic
func.func @selected_all_parallel_tile_fallback_for_broadcast_transpose() {
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
