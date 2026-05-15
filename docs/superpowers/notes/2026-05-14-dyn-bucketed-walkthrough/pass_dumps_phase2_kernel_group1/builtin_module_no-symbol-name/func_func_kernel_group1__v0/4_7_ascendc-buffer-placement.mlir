// -----// IR Dump After AscendCBufferPlacementPass (ascendc-buffer-placement) //----- //
func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg4: index {vector_plan.default_tile_size = 128 : i64}, %arg5: index {vector_plan.default_tile_size = 16 : i64}) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg3, %c0 : memref<?x?xf32>
  %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
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
    %dim_6 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
    %alloc_7 = memref.alloc(%arg5, %dim_6) : memref<?x?xf32, 9>
    %dim_8 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
    %alloc_9 = memref.alloc(%arg5, %dim_8) : memref<?x?xf32, 9>
    %dim_10 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
    %alloc_11 = memref.alloc(%arg5, %dim_10) : memref<?x?xf32, 9>
    %alloc_12 = memref.alloc(%arg5) : memref<?xf32, 10>
    %6 = scf.for %arg8 = %c0 to %5 step %arg5 iter_args(%arg9 = %arg7) -> (memref<?xf32>) {
      %9 = arith.addi %arg6, %arg8 : index
      %dim_13 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%9, 0] [%arg5, %dim_13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_14 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_15 = memref.subview %collapse_shape_1[%9, 0] [%arg5, %dim_14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_16 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_17 = memref.subview %collapse_shape_2[%9, 0] [%arg5, %dim_16] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_18 = memref.subview %arg9[%9] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %alloc_7 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      memref.copy %subview_15, %alloc_9 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      memref.copy %subview_17, %alloc_11 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%alloc_7, %alloc_9, %alloc_11 : memref<?x?xf32, 9>, memref<?x?xf32, 9>, memref<?x?xf32, 9>) outs(%alloc_12 : memref<?xf32, 10>) {
      ^bb0(%in: f32, %in_19: f32, %in_20: f32, %out: f32):
        %10 = arith.addf %in, %in_19 : f32
        %11 = arith.mulf %10, %in_20 : f32
        %12 = arith.addf %out, %11 : f32
        linalg.yield %12 : f32
      }
      memref.copy %alloc_12, %subview_18 : memref<?xf32, 10> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %arg9 : memref<?xf32>
    }
    %7 = arith.cmpi slt, %5, %3 : index
    %8 = scf.if %7 -> (memref<?xf32>) {
      %9 = arith.subi %0, %arg5 : index
      %dim_13 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%9, 0] [%arg5, %dim_13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_14 = memref.dim %collapse_shape_1, %c1 : memref<?x?xf32>
      %subview_15 = memref.subview %collapse_shape_1[%9, 0] [%arg5, %dim_14] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_16 = memref.dim %collapse_shape_2, %c1 : memref<?x?xf32>
      %subview_17 = memref.subview %collapse_shape_2[%9, 0] [%arg5, %dim_16] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_18 = memref.subview %6[%9] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %alloc_19 = memref.alloc(%arg5, %dim_13) : memref<?x?xf32, 9>
      memref.copy %subview, %alloc_19 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      %alloc_20 = memref.alloc(%arg5, %dim_14) : memref<?x?xf32, 9>
      memref.copy %subview_15, %alloc_20 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      %alloc_21 = memref.alloc(%arg5, %dim_16) : memref<?x?xf32, 9>
      memref.copy %subview_17, %alloc_21 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, 9>
      %alloc_22 = memref.alloc(%arg5) : memref<?xf32, 10>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%alloc_19, %alloc_20, %alloc_21 : memref<?x?xf32, 9>, memref<?x?xf32, 9>, memref<?x?xf32, 9>) outs(%alloc_22 : memref<?xf32, 10>) {
      ^bb0(%in: f32, %in_23: f32, %in_24: f32, %out: f32):
        %10 = arith.addf %in, %in_23 : f32
        %11 = arith.mulf %10, %in_24 : f32
        %12 = arith.addf %out, %11 : f32
        linalg.yield %12 : f32
      }
      memref.copy %alloc_22, %subview_18 : memref<?xf32, 10> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %6 : memref<?xf32>
    } else {
      scf.yield %6 : memref<?xf32>
    }
    scf.yield %8 : memref<?xf32>
  }
  %expand_shape = memref.expand_shape %1 [[0, 1]] output_shape [%dim, %dim_0] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

