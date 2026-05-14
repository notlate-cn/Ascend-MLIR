// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s
// RUN: afir-opt %s --ascend-compute-lower --ascend-parallelize | FileCheck %s --check-prefix=PARALLELIZE

#identity = affine_map<(d0, d1) -> (d0, d1)>
#project_row = affine_map<(d0, d1) -> (d0)>
#project_col = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @selected_all_parallel_tile_materializes_loop
// CHECK: scf.for %{{.*}} = %c0 to %c70 step %c64
// CHECK: arith.minsi
// CHECK: ascendc.add_l2
// CHECK: memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, %c128] [1, 1] : memref<70x128xf16>
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: linalg.generic

// PARALLELIZE-LABEL: func.func @selected_all_parallel_tile_materializes_loop
// PARALLELIZE: ascendc.get_block_idx
// PARALLELIZE: arith.muli %{{.*}}, %c64
// PARALLELIZE: scf.if
// PARALLELIZE: ascendc.add_l2
// PARALLELIZE: memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, %c128] [1, 1] : memref<70x128xf16>
// PARALLELIZE: ascendc.data_copy_l2
// PARALLELIZE-NOT: scf.for
// PARALLELIZE-NOT: linalg.generic
func.func @selected_all_parallel_tile_materializes_loop() {
  %a = memref.alloc() : memref<70x128xf16, 9 : i32>
  %b = memref.alloc() : memref<70x128xf16, 9 : i32>
  %out = memref.alloc() : memref<70x128xf16, 10 : i32>
  %gm = memref.alloc() : memref<70x128xf16>
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
  memref.copy %out, %gm : memref<70x128xf16, 10 : i32> to memref<70x128xf16>
  return
}

// CHECK-LABEL: func.func @selected_all_parallel_tile_materializes_projected_inner
// CHECK: ascendc.pipe.init_queue %{{.*}}, %{{.*}}, %c1_i32, %c256
// CHECK: scf.for %{{.*}} = %c0 to %c70 step %c64
// CHECK: ascendc.broadcast_l2 %{{.*}}, %{{.*}}, %{{.*}}, %c128_i32, %{{.*}}, %c1_i32
// CHECK: ascendc.broadcast_l2 %{{.*}}, %{{.*}}, %{{.*}}, %c128_i32, %c1_i32, %c128_i32
// CHECK: ascendc.mul_l2
// CHECK: memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, %c128] [1, 1] : memref<70x128xf16>
// CHECK-NOT: linalg.generic

// PARALLELIZE-LABEL: func.func @selected_all_parallel_tile_materializes_projected_inner
// PARALLELIZE: ascendc.pipe.init_queue %{{.*}}, %{{.*}}, %c1_i32, %c256
// PARALLELIZE: ascendc.get_block_idx
// PARALLELIZE: ascendc.broadcast_l2 %{{.*}}, %{{.*}}, %{{.*}}, %c128_i32, %{{.*}}, %c1_i32
// PARALLELIZE: ascendc.broadcast_l2 %{{.*}}, %{{.*}}, %{{.*}}, %c128_i32, %c1_i32, %c128_i32
// PARALLELIZE: ascendc.mul_l2
// PARALLELIZE-NOT: linalg.generic
func.func @selected_all_parallel_tile_materializes_projected_inner() {
  %a = memref.alloc() : memref<70x128xf16, 9 : i32>
  %row = memref.alloc() : memref<70xf16, 9 : i32>
  %col = memref.alloc() : memref<128xf16, 9 : i32>
  %out = memref.alloc() : memref<70x128xf16, 10 : i32>
  %gm = memref.alloc() : memref<70x128xf16>
  linalg.generic {
      indexing_maps = [#identity, #project_row, #project_col, #identity],
      iterator_types = ["parallel", "parallel"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>}
      ins(%a, %row, %col : memref<70x128xf16, 9 : i32>,
                            memref<70xf16, 9 : i32>,
                            memref<128xf16, 9 : i32>)
      outs(%out : memref<70x128xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %row_elem: f16, %col_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %row_elem : f16
      %scaled = arith.mulf %sum, %col_elem : f16
      linalg.yield %scaled : f16
  }
  memref.copy %out, %gm : memref<70x128xf16, 10 : i32> to memref<70x128xf16>
  return
}

// CHECK-LABEL: func.func @selected_all_parallel_tile_allows_shared_read_subviews
// CHECK-COUNT-2: scf.for
// CHECK-NOT: linalg.generic

// PARALLELIZE-LABEL: func.func @selected_all_parallel_tile_allows_shared_read_subviews
// PARALLELIZE-COUNT-2: ascendc.get_block_idx
// PARALLELIZE-NOT: linalg.generic
func.func @selected_all_parallel_tile_allows_shared_read_subviews(
    %input: memref<140x128xf16, 9 : i32>,
    %row0: memref<70xf16, 9 : i32>,
    %row1: memref<70xf16, 9 : i32>,
    %col0: memref<128xf16, 9 : i32>,
    %col1: memref<128xf16, 9 : i32>,
    %gm: memref<140x128xf16>) {
  %c0 = arith.constant 0 : index
  %c70 = arith.constant 70 : index
  %c128 = arith.constant 128 : index
  %top = memref.subview %input[0, 0] [70, 128] [1, 1]
      : memref<140x128xf16, 9 : i32> to memref<70x128xf16, strided<[128, 1]>, 9 : i32>
  %bottom = memref.subview %input[70, 0] [70, 128] [1, 1]
      : memref<140x128xf16, 9 : i32> to memref<70x128xf16, strided<[128, 1], offset: 8960>, 9 : i32>
  %out0 = memref.alloc() : memref<70x128xf16, 10 : i32>
  linalg.generic {
      indexing_maps = [#identity, #project_row, #project_col, #identity],
      iterator_types = ["parallel", "parallel"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>}
      ins(%top, %row0, %col0 : memref<70x128xf16, strided<[128, 1]>, 9 : i32>,
                                memref<70xf16, 9 : i32>,
                                memref<128xf16, 9 : i32>)
      outs(%out0 : memref<70x128xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %row_elem: f16, %col_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %row_elem : f16
      %scaled = arith.mulf %sum, %col_elem : f16
      linalg.yield %scaled : f16
  }
  %out1 = memref.alloc() : memref<70x128xf16, 10 : i32>
  linalg.generic {
      indexing_maps = [#identity, #project_row, #project_col, #identity],
      iterator_types = ["parallel", "parallel"],
      ascend.schedule.selected_tile_shape = array<i64: 64, 128>}
      ins(%bottom, %row1, %col1 : memref<70x128xf16, strided<[128, 1], offset: 8960>, 9 : i32>,
                                   memref<70xf16, 9 : i32>,
                                   memref<128xf16, 9 : i32>)
      outs(%out1 : memref<70x128xf16, 10 : i32>) {
    ^bb0(%a_elem: f16, %row_elem: f16, %col_elem: f16, %acc: f16):
      %sum = arith.addf %a_elem, %row_elem : f16
      %scaled = arith.mulf %sum, %col_elem : f16
      linalg.yield %scaled : f16
  }
  %dst0 = memref.subview %gm[0, 0] [70, 128] [1, 1]
      : memref<140x128xf16> to memref<70x128xf16, strided<[128, 1]>>
  memref.copy %out0, %dst0 : memref<70x128xf16, 10 : i32> to memref<70x128xf16, strided<[128, 1]>>
  %dst1 = memref.subview %gm[%c70, %c0] [%c70, %c128] [1, 1]
      : memref<140x128xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
  memref.copy %out1, %dst1 : memref<70x128xf16, 10 : i32> to memref<?x?xf16, strided<[128, 1], offset: ?>>
  return
}
