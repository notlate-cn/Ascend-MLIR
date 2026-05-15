// -----// IR Dump After VectorPlanFoldShadowAlloc (vector-plan-fold-shadow-alloc) //----- //
func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg4: index {vector_plan.default_tile_size = 128 : i64}, %arg5: index {vector_plan.default_tile_size = 16 : i64}) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg3, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  memref.copy %arg3, %alloc : memref<?x?xf32> to memref<?x?xf32>
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_2 = memref.collapse_shape %arg2 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_3 = memref.collapse_shape %alloc [[0, 1]] : memref<?x?xf32> into memref<?xf32>
  %dim_4 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim_5 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %0 = arith.muli %dim_4, %dim_5 : index
  %1 = scf.for %arg6 = %c0 to %0 step %arg4 iter_args(%arg7 = %collapse_shape_3) -> (memref<?xf32>) {
    %2 = arith.subi %0, %arg6 : index
    %3 = arith.minsi %arg4, %2 : index
    %4 = arith.divsi %3, %arg5 : index
    %5 = arith.muli %4, %arg5 : index
    %6 = scf.for %arg8 = %c0 to %5 step %arg5 iter_args(%arg9 = %arg7) -> (memref<?xf32>) {
      %9 = arith.addi %arg6, %arg8 : index
      %dim_6 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%9, 0] [%arg5, %dim_6] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_7 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_8 = memref.subview %collapse_shape_1[%9, 0] [%arg5, %dim_7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_9 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_10 = memref.subview %collapse_shape_2[%9, 0] [%arg5, %dim_9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_11 = memref.subview %arg9[%9] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%subview, %subview_8, %subview_10 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_11 : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_12: f32, %in_13: f32, %out: f32):
        %10 = arith.addf %in, %in_12 : f32
        %11 = arith.mulf %10, %in_13 : f32
        %12 = arith.addf %out, %11 : f32
        linalg.yield %12 : f32
      }
      memref.copy %subview_11, %subview_11 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %arg9 : memref<?xf32>
    }
    %7 = arith.cmpi slt, %5, %3 : index
    %8 = scf.if %7 -> (memref<?xf32>) {
      %9 = arith.subi %0, %arg5 : index
      %dim_6 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%9, 0] [%arg5, %dim_6] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_7 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_8 = memref.subview %collapse_shape_1[%9, 0] [%arg5, %dim_7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_9 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_10 = memref.subview %collapse_shape_2[%9, 0] [%arg5, %dim_9] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_11 = memref.subview %6[%9] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%subview, %subview_8, %subview_10 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_11 : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_12: f32, %in_13: f32, %out: f32):
        %10 = arith.addf %in, %in_12 : f32
        %11 = arith.mulf %10, %in_13 : f32
        %12 = arith.addf %out, %11 : f32
        linalg.yield %12 : f32
      }
      memref.copy %subview_11, %subview_11 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %6 : memref<?xf32>
    } else {
      scf.yield %6 : memref<?xf32>
    }
    scf.yield %8 : memref<?xf32>
  } {ascendc.parallel}
  %expand_shape = memref.expand_shape %1 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

