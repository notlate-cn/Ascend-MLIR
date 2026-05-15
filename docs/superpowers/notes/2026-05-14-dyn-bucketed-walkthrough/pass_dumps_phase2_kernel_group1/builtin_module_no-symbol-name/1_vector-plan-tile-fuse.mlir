// -----// IR Dump After VectorPlanTileFuse (vector-plan-tile-fuse) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 4 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group1__v0"}]} {
  func.func private @kernel_group1__v0(%arg0: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: tensor<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg4: index {vector_plan.default_tile_size = 128 : i64}, %arg5: index {vector_plan.default_tile_size = 16 : i64}) -> tensor<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %0 = arith.muli %c1, %dim : index
    %c1_0 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c1_0 : tensor<?x?x?xf32>
    %1 = arith.muli %0, %dim_1 : index
    %2 = arith.ceildivsi %1, %arg4 : index
    %c1_2 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim_3 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %3 = arith.muli %c1_2, %dim_3 : index
    %4 = bufferization.alloc_tensor() copy(%arg3) {afir.symbolic_shapes = ["s0,s1"]} : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_4 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_5 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_6 = tensor.collapse_shape %4 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
    %c0_7 = arith.constant 0 : index
    %c1_8 = arith.constant 1 : index
    %c0_9 = arith.constant 0 : index
    %dim_10 = tensor.dim %arg0, %c0_9 : tensor<?x?x?xf32>
    %5 = arith.muli %c1_8, %dim_10 : index
    %c1_11 = arith.constant 1 : index
    %dim_12 = tensor.dim %arg0, %c1_11 : tensor<?x?x?xf32>
    %6 = arith.muli %5, %dim_12 : index
    %7 = scf.for %arg6 = %c0_7 to %6 step %arg4 iter_args(%arg7 = %collapsed_6) -> (tensor<?xf32>) {
      %c1_17 = arith.constant 1 : index
      %c0_18 = arith.constant 0 : index
      %dim_19 = tensor.dim %arg0, %c0_18 : tensor<?x?x?xf32>
      %8 = arith.muli %c1_17, %dim_19 : index
      %c1_20 = arith.constant 1 : index
      %dim_21 = tensor.dim %arg0, %c1_20 : tensor<?x?x?xf32>
      %9 = arith.muli %8, %dim_21 : index
      %10 = arith.subi %9, %arg6 : index
      %11 = arith.minsi %arg4, %10 : index
      %12 = arith.divsi %11, %arg5 : index
      %13 = arith.muli %12, %arg5 : index
      %14 = scf.for %arg8 = %c0_7 to %13 step %arg5 iter_args(%arg9 = %arg7) -> (tensor<?xf32>) {
        %17 = arith.addi %arg6, %arg8 : index
        %c0_22 = arith.constant 0 : index
        %c1_23 = arith.constant 1 : index
        %c1_24 = arith.constant 1 : index
        %dim_25 = tensor.dim %collapsed, %c1_24 : tensor<?x?xf32>
        %extracted_slice = tensor.extract_slice %collapsed[%17, %c0_22] [%arg5, %dim_25] [%c1_23, %c1_23] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_26 = arith.constant 0 : index
        %c1_27 = arith.constant 1 : index
        %c1_28 = arith.constant 1 : index
        %dim_29 = tensor.dim %collapsed_4, %c1_28 : tensor<?x?xf32>
        %extracted_slice_30 = tensor.extract_slice %collapsed_4[%17, %c0_26] [%arg5, %dim_29] [%c1_27, %c1_27] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_31 = arith.constant 0 : index
        %c1_32 = arith.constant 1 : index
        %c1_33 = arith.constant 1 : index
        %dim_34 = tensor.dim %collapsed_5, %c1_33 : tensor<?x?xf32>
        %extracted_slice_35 = tensor.extract_slice %collapsed_5[%17, %c0_31] [%arg5, %dim_34] [%c1_32, %c1_32] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_36 = arith.constant 0 : index
        %c1_37 = arith.constant 1 : index
        %extracted_slice_38 = tensor.extract_slice %arg9[%17] [%arg5] [%c1_37] : tensor<?xf32> to tensor<?xf32>
        %18 = linalg.generic {indexing_maps = [#map, #map, #map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_30, %extracted_slice_35 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_38 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_41: f32, %in_42: f32, %out: f32):
          %19 = arith.addf %in, %in_41 : f32
          %20 = arith.mulf %19, %in_42 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<?xf32>
        %c0_39 = arith.constant 0 : index
        %c1_40 = arith.constant 1 : index
        %inserted_slice = tensor.insert_slice %18 into %arg9[%17] [%arg5] [%c1_40] : tensor<?xf32> into tensor<?xf32>
        scf.yield %inserted_slice : tensor<?xf32>
      }
      %15 = arith.cmpi slt, %13, %11 : index
      %16 = scf.if %15 -> (tensor<?xf32>) {
        %17 = arith.subi %9, %arg5 : index
        %c0_22 = arith.constant 0 : index
        %c1_23 = arith.constant 1 : index
        %c1_24 = arith.constant 1 : index
        %dim_25 = tensor.dim %collapsed, %c1_24 : tensor<?x?xf32>
        %extracted_slice = tensor.extract_slice %collapsed[%17, %c0_22] [%arg5, %dim_25] [%c1_23, %c1_23] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_26 = arith.constant 0 : index
        %c1_27 = arith.constant 1 : index
        %c1_28 = arith.constant 1 : index
        %dim_29 = tensor.dim %collapsed_4, %c1_28 : tensor<?x?xf32>
        %extracted_slice_30 = tensor.extract_slice %collapsed_4[%17, %c0_26] [%arg5, %dim_29] [%c1_27, %c1_27] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_31 = arith.constant 0 : index
        %c1_32 = arith.constant 1 : index
        %c1_33 = arith.constant 1 : index
        %dim_34 = tensor.dim %collapsed_5, %c1_33 : tensor<?x?xf32>
        %extracted_slice_35 = tensor.extract_slice %collapsed_5[%17, %c0_31] [%arg5, %dim_34] [%c1_32, %c1_32] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_36 = arith.constant 0 : index
        %c1_37 = arith.constant 1 : index
        %extracted_slice_38 = tensor.extract_slice %14[%17] [%arg5] [%c1_37] : tensor<?xf32> to tensor<?xf32>
        %18 = linalg.generic {indexing_maps = [#map, #map, #map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_30, %extracted_slice_35 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_38 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_41: f32, %in_42: f32, %out: f32):
          %19 = arith.addf %in, %in_41 : f32
          %20 = arith.mulf %19, %in_42 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<?xf32>
        %c0_39 = arith.constant 0 : index
        %c1_40 = arith.constant 1 : index
        %inserted_slice = tensor.insert_slice %18 into %14[%17] [%arg5] [%c1_40] : tensor<?xf32> into tensor<?xf32>
        scf.yield %inserted_slice : tensor<?xf32>
      } else {
        scf.yield %14 : tensor<?xf32>
      }
      scf.yield %16 : tensor<?xf32>
    } {ascendc.parallel}
    %c0_13 = arith.constant 0 : index
    %dim_14 = tensor.dim %4, %c0_13 : tensor<?x?xf32>
    %c1_15 = arith.constant 1 : index
    %dim_16 = tensor.dim %4, %c1_15 : tensor<?x?xf32>
    %expanded = tensor.expand_shape %7 [[0, 1]] output_shape [%dim_14, %dim_16] : tensor<?xf32> into tensor<?x?xf32>
    return %expanded : tensor<?x?xf32>
  }
}


