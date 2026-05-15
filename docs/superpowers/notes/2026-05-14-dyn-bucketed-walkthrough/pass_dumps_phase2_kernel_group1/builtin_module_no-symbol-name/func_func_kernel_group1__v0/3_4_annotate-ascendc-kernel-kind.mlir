// -----// IR Dump After AnnotateAscendCKernelKindPass (annotate-ascendc-kernel-kind) //----- //
func.func private @kernel_group1__v0(%arg0: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: memref<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: memref<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg4: index {vector_plan.default_tile_size = 128 : i64}, %arg5: index {vector_plan.default_tile_size = 16 : i64}) -> memref<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %dim = memref.dim %arg3, %c0_0 : memref<?x?xf32>
  %c1_1 = arith.constant 1 : index
  %dim_2 = memref.dim %arg3, %c1_1 : memref<?x?xf32>
  %alloc = memref.alloc(%dim, %dim_2) {alignment = 64 : i64} : memref<?x?xf32>
  memref.copy %arg3, %alloc : memref<?x?xf32> to memref<?x?xf32>
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_3 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_4 = memref.collapse_shape %arg2 [[0, 1], [2]] : memref<?x?x?xf32> into memref<?x?xf32>
  %collapse_shape_5 = memref.collapse_shape %alloc [[0, 1]] : memref<?x?xf32> into memref<?xf32>
  %dim_6 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim_7 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %0 = arith.muli %dim_6, %dim_7 : index
  %1 = scf.for %arg6 = %c0 to %0 step %arg4 iter_args(%arg7 = %collapse_shape_5) -> (memref<?xf32>) {
    %dim_10 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
    %dim_11 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
    %2 = arith.muli %dim_10, %dim_11 : index
    %3 = arith.subi %2, %arg6 : index
    %4 = arith.minsi %arg4, %3 : index
    %5 = arith.divsi %4, %arg5 : index
    %6 = arith.muli %5, %arg5 : index
    %7 = scf.for %arg8 = %c0 to %6 step %arg5 iter_args(%arg9 = %arg7) -> (memref<?xf32>) {
      %10 = arith.addi %arg6, %arg8 : index
      %dim_12 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%10, 0] [%arg5, %dim_12] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_13 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
      %subview_14 = memref.subview %collapse_shape_3[%10, 0] [%arg5, %dim_13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_15 = memref.dim %collapse_shape_4, %c1 : memref<?x?xf32>
      %subview_16 = memref.subview %collapse_shape_4[%10, 0] [%arg5, %dim_15] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_17 = memref.subview %arg9[%10] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%subview, %subview_14, %subview_16 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_17 : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_19: f32, %in_20: f32, %out: f32):
        %11 = arith.addf %in, %in_19 : f32
        %12 = arith.mulf %11, %in_20 : f32
        %13 = arith.addf %out, %12 : f32
        linalg.yield %13 : f32
      }
      %subview_18 = memref.subview %arg9[%10] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview_17, %subview_18 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %arg9 : memref<?xf32>
    }
    %8 = arith.cmpi slt, %6, %4 : index
    %9 = scf.if %8 -> (memref<?xf32>) {
      %10 = arith.subi %2, %arg5 : index
      %dim_12 = memref.dim %collapse_shape, %c1 : memref<?x?xf32>
      %subview = memref.subview %collapse_shape[%10, 0] [%arg5, %dim_12] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_13 = memref.dim %collapse_shape_3, %c1 : memref<?x?xf32>
      %subview_14 = memref.subview %collapse_shape_3[%10, 0] [%arg5, %dim_13] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %dim_15 = memref.dim %collapse_shape_4, %c1 : memref<?x?xf32>
      %subview_16 = memref.subview %collapse_shape_4[%10, 0] [%arg5, %dim_15] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_17 = memref.subview %7[%10] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%subview, %subview_14, %subview_16 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_17 : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%in: f32, %in_19: f32, %in_20: f32, %out: f32):
        %11 = arith.addf %in, %in_19 : f32
        %12 = arith.mulf %11, %in_20 : f32
        %13 = arith.addf %out, %12 : f32
        linalg.yield %13 : f32
      }
      %subview_18 = memref.subview %7[%10] [%arg5] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview_17, %subview_18 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      scf.yield %7 : memref<?xf32>
    } else {
      scf.yield %7 : memref<?xf32>
    }
    scf.yield %9 : memref<?xf32>
  } {ascendc.parallel}
  %dim_8 = memref.dim %arg3, %c0 : memref<?x?xf32>
  %dim_9 = memref.dim %arg3, %c1 : memref<?x?xf32>
  %expand_shape = memref.expand_shape %1 [[0, 1]] output_shape [%dim_8, %dim_9] : memref<?xf32> into memref<?x?xf32>
  return %expand_shape : memref<?x?xf32>
}

