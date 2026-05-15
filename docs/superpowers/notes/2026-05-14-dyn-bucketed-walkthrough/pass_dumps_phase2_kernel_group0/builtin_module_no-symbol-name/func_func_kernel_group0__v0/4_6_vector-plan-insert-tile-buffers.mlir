// -----// IR Dump After VectorPlanInsertTileBuffers (vector-plan-insert-tile-buffers) //----- //
func.func private @kernel_group0__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg4: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg5: index {vector_plan.default_tile_size = 128 : i64}, %arg6: index {vector_plan.default_tile_size = 16 : i64}) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg4, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg4, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_1 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_2 = memref.collapse_shape %arg2 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_3 = memref.collapse_shape %arg3 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_4 = memref.collapse_shape %alloc [[0, 1]] : memref<?x?xf32> into memref<?xf32>
  %dim_5 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim_6 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %0 = arith.muli %dim_5, %dim_6 : index
  %1 = scf.for %arg7 = %c0 to %0 step %arg5 iter_args(%arg8 = %collapse_shape_4) -> (memref<?xf32>) {
    %2 = arith.subi %0, %arg7 : index
    %3 = arith.minsi %arg5, %2 : index
    %4 = arith.divsi %3, %arg6 : index
    %5 = arith.muli %4, %arg6 : index
    %dim_7 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
    %alloc_8 = memref.alloc(%arg6, %dim_7) : memref<?x?xf32, 9>
    %dim_9 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
    %alloc_10 = memref.alloc(%arg6, %dim_9) : memref<?x?xf32, 9>
    %dim_11 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
    %alloc_12 = memref.alloc(%arg6, %dim_11) : memref<?x?xf32, 9>
    %dim_13 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
    %alloc_14 = memref.alloc(%arg6, %dim_13) : memref<?x?xf32, 9>
    %alloc_15 = memref.alloc(%arg6) : memref<?xf32, 10>
    %6 = scf.for %arg9 = %c0 to %5 step %arg6 iter_args(%arg10 = %arg8) -> (memref<?xf32>) {
      %9 = arith.addi %arg7, %arg9 : index
      %dim_16 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%9, 0] [%arg6, %dim_16] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_17 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_18 = memref.subview %collapse_shape_1[%9, 0] [%arg6, %dim_17] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_19 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_20 = memref.subview %collapse_shape_2[%9, 0] [%arg6, %dim_19] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_21 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
      %subview_22 = memref.subview %collapse_shape_3[%9, 0] [%arg6, %dim_21] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_23 = memref.subview %arg10[%9] [%arg6] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      memref.copy %subview_18, %alloc_10 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      memref.copy %subview_20, %alloc_12 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      memref.copy %subview_22, %alloc_14 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%alloc_8, %alloc_10, %alloc_12, %alloc_14 : memref<?x?xf32, 9>, memref<?x?xf32, 9>, memref<?x?xf32, 9>, memref<?x?xf32, 9>) outs(%alloc_15 : memref<?xf32, 10>) {
      ^bb0(%in: f32, %in_24: f32, %in_25: f32, %in_26: f32, %out: f32):
        %10 = arith.addf %in, %in_24 : f32
        %11 = arith.mulf %10, %in_25 : f32
        %12 = arith.addf %11, %in_26 : f32
        %13 = arith.addf %out, %12 : f32
        linalg.yield %13 : f32
      }
      memref.copy %alloc_15, %subview_23 : memref<?xf32, 10> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %arg10 : memref<?xf32>
    }
    %7 = arith.cmpi slt, %5, %3 : index
    %8 = scf.if %7 -> (memref<?xf32>) {
      %9 = arith.subi %0, %arg6 : index
      %dim_16 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%9, 0] [%arg6, %dim_16] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_17 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_18 = memref.subview %collapse_shape_1[%9, 0] [%arg6, %dim_17] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_19 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_20 = memref.subview %collapse_shape_2[%9, 0] [%arg6, %dim_19] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_21 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
      %subview_22 = memref.subview %collapse_shape_3[%9, 0] [%arg6, %dim_21] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_23 = memref.subview %6[%9] [%arg6] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %alloc_24 = memref.alloc(%arg6, %dim_16) : memref<?x?xf32, 9>
      memref.copy %subview, %alloc_24 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      %alloc_25 = memref.alloc(%arg6, %dim_17) : memref<?x?xf32, 9>
      memref.copy %subview_18, %alloc_25 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      %alloc_26 = memref.alloc(%arg6, %dim_19) : memref<?x?xf32, 9>
      memref.copy %subview_20, %alloc_26 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      %alloc_27 = memref.alloc(%arg6, %dim_21) : memref<?x?xf32, 9>
      memref.copy %subview_22, %alloc_27 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      %alloc_28 = memref.alloc(%arg6) : memref<?xf32, 10>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%alloc_24, %alloc_25, %alloc_26, %alloc_27 : memref<?x?xf32, 9>, memref<?x?xf32, 9>, memref<?x?xf32, 9>, memref<?x?xf32, 9>) outs(%alloc_28 : memref<?xf32, 10>) {
      ^bb0(%in: f32, %in_29: f32, %in_30: f32, %in_31: f32, %out: f32):
        %10 = arith.addf %in, %in_29 : f32
        %11 = arith.mulf %10, %in_30 : f32
        %12 = arith.addf %11, %in_31 : f32
        %13 = arith.addf %out, %12 : f32
        linalg.yield %13 : f32
      }
      memref.copy %alloc_28, %subview_23 : memref<?xf32, 10> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %6 : memref<?xf32>
    } else {
      scf.yield %6 : memref<?xf32>
    }
    scf.yield %8 : memref<?xf32>
  } {ascendc.parallel}
  %expand_shape = memref.expand_shape %1 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

