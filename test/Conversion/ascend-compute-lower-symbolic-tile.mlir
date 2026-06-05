// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>
#project_row = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @symbolic_all_parallel_tile_step(
// CHECK-SAME: %[[TILE_M:.*]]: i64
// CHECK: %[[STEP:.*]] = arith.index_cast %[[TILE_M]] : i64 to index
// CHECK: scf.for %{{.*}} = %c0 to %c70 step %[[STEP]]
// CHECK-NOT: linalg.generic
func.func @symbolic_all_parallel_tile_step() {
  %a = memref.alloc() : memref<70x128xf16, 9 : i32>
  %b = memref.alloc() : memref<70x128xf16, 9 : i32>
  %out = memref.alloc() : memref<70x128xf16, 10 : i32>
  %gm = memref.alloc() : memref<70x128xf16>
  linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"],
      ascend.schedule.tile_binding = "symbolic",
      ascend.schedule.tile_params = [
        {
          axis = 0 : i64,
          axis_kind = "parallel",
          binding = "runtime",
          default = 64 : i64,
          extent = 70 : i64,
          name = "TB_M",
          primitive_uses = ["data_copy", "vector_compute", "write_back"],
          roles = ["bind_core", "kernel_loop", "vectorize"],
          upper_bound = 64 : i64
        },
        {
          axis = 1 : i64,
          axis_kind = "parallel",
          binding = "extent",
          default = 128 : i64,
          extent = 128 : i64,
          name = "TB_N",
          primitive_uses = ["data_copy", "vector_compute", "write_back"],
          roles = ["bind_core", "kernel_loop", "vectorize"],
          upper_bound = 128 : i64
        }
      ] }
      ins(%a, %b : memref<70x128xf16, 9 : i32>,
                    memref<70x128xf16, 9 : i32>)
      outs(%out : memref<70x128xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %b_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %b_elem : f16
      linalg.yield %sum : f16
  }
  memref.copy %out, %gm : memref<70x128xf16, 10 : i32> to memref<70x128xf16>
  return
}

// CHECK-LABEL: func.func @symbolic_reduction_tile_step(
// CHECK-SAME: %[[RED_TILE_M:.*]]: i64
// CHECK: %[[RED_STEP:.*]] = arith.index_cast %[[RED_TILE_M]] : i64 to index
// CHECK: scf.for %{{.*}} = %c0 to %c70 step %[[RED_STEP]]
// CHECK-NOT: linalg.generic
func.func @symbolic_reduction_tile_step() {
  %a = memref.alloc() : memref<70x128xf16, 9 : i32>
  %out = memref.alloc() : memref<70xf16, 10 : i32>
  %gm = memref.alloc() : memref<70xf16>
  linalg.generic {
      indexing_maps = [#identity, #project_row],
      iterator_types = ["parallel", "reduction"],
      ascend.schedule.tile_binding = "symbolic",
      ascend.schedule.tile_params = [
        {
          axis = 0 : i64,
          axis_kind = "parallel",
          binding = "runtime",
          default = 32 : i64,
          extent = 70 : i64,
          name = "TB_M",
          primitive_uses = ["data_copy", "vector_compute", "write_back"],
          roles = ["bind_core", "kernel_loop", "vectorize"],
          upper_bound = 32 : i64
        },
        {
          axis = 1 : i64,
          axis_kind = "reduction",
          binding = "extent",
          default = 128 : i64,
          extent = 128 : i64,
          name = "TB_N",
          primitive_uses = ["reduction"],
          roles = ["full_reduction"],
          upper_bound = 128 : i64
        }
      ] }
      ins(%a : memref<70x128xf16, 9 : i32>)
      outs(%out : memref<70xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %acc : f16
      linalg.yield %sum : f16
  }
  memref.copy %out, %gm : memref<70xf16, 10 : i32> to memref<70xf16>
  return
}
