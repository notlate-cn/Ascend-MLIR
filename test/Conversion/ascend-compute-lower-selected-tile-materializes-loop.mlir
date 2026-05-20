// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#project_parallel = affine_map<(d0, d1) -> (d0)>
#full = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @selected_tile_materializes_loop
// CHECK: scf.for %{{.*}} = %c0 to %c70 step %c64
// CHECK: arith.minsi
// CHECK: ascendc.pipe.init_buffer
// CHECK: ascendc.broadcast_l2
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK: memref.subview %{{.*}}[%{{.*}}] [%{{.*}}] [1] : memref<70xf16>
// CHECK: emitasc.verbatim
// CHECK-NOT: linalg.generic
func.func @selected_tile_materializes_loop() {
  %a = memref.alloc() : memref<70xf16, 9 : i32>
  %b = memref.alloc() : memref<70x128xf16, 9 : i32>
  %out = memref.alloc() : memref<70xf16, 10 : i32>
  %gm = memref.alloc() : memref<70xf16>
  linalg.generic {
      indexing_maps = [#project_parallel, #full, #project_parallel],
      iterator_types = ["parallel", "reduction"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>}
      ins(%a, %b : memref<70xf16, 9 : i32>, memref<70x128xf16, 9 : i32>)
      outs(%out : memref<70xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %b_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %b_elem : f16
      %next = arith.addf %acc, %sum : f16
      linalg.yield %next : f16
  }
  memref.copy %out, %gm : memref<70xf16, 10 : i32> to memref<70xf16>
  return
}

// CHECK-LABEL: func.func @selected_reduction_erases_redundant_zero_fill
// CHECK: memref.alloc
// CHECK-NOT: ascendc.data_copy_l2
// CHECK: scf.for
// CHECK: ascendc.duplicate_l2
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK-NOT: linalg.fill
func.func @selected_reduction_erases_redundant_zero_fill(
    %a: memref<?xf16>, %b: memref<?x?xf16>) -> memref<?xf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f16
  %m = memref.dim %a, %c0 : memref<?xf16>
  %gm = memref.alloc(%m) : memref<?xf16>
  linalg.fill ins(%zero : f16) outs(%gm : memref<?xf16>)
  %out = memref.alloc(%m) : memref<?xf16, 10 : i32>
  linalg.generic {
      indexing_maps = [#project_parallel, #full, #project_parallel],
      iterator_types = ["parallel", "reduction"],
      ascend.schedule.selected_tile_shape = array<i64: 32, -9223372036854775808>}
      ins(%a, %b : memref<?xf16>, memref<?x?xf16>)
      outs(%out : memref<?xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %b_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %b_elem : f16
      %next = arith.addf %acc, %sum : f16
      linalg.yield %next : f16
  }
  memref.copy %out, %gm : memref<?xf16, 10 : i32> to memref<?xf16>
  return %gm : memref<?xf16>
}
