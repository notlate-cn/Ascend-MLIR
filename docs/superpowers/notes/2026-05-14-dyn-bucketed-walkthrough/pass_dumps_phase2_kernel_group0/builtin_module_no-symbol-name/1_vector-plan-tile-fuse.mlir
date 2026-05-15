// -----// IR Dump After VectorPlanTileFuse (vector-plan-tile-fuse) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 6 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group0__v0"}]} {
  func.func private @kernel_group0__v0(%arg0: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg4: tensor<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg5: index {vector_plan.default_tile_size = 128 : i64}, %arg6: index {vector_plan.default_tile_size = 16 : i64}) -> tensor<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %0 = arith.muli %c1, %dim : index
    %c1_0 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c1_0 : tensor<?x?x?xf32>
    %1 = arith.muli %0, %dim_1 : index
    %2 = arith.ceildivsi %1, %arg5 : index
    %c1_2 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim_3 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %3 = arith.muli %c1_2, %dim_3 : index
    %4 = bufferization.alloc_tensor() copy(%arg4) {afir.symbolic_shapes = ["s0,s1"]} : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_4 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_5 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_6 = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_7 = tensor.collapse_shape %4 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
    %c0_8 = arith.constant 0 : index
    %c1_9 = arith.constant 1 : index
    %c0_10 = arith.constant 0 : index
    %dim_11 = tensor.dim %arg0, %c0_10 : tensor<?x?x?xf32>
    %5 = arith.muli %c1_9, %dim_11 : index
    %c1_12 = arith.constant 1 : index
    %dim_13 = tensor.dim %arg0, %c1_12 : tensor<?x?x?xf32>
    %6 = arith.muli %5, %dim_13 : index
    %7 = scf.for %arg7 = %c0_8 to %6 step %arg5 iter_args(%arg8 = %collapsed_7) -> (tensor<?xf32>) {
      %c1_18 = arith.constant 1 : index
      %c0_19 = arith.constant 0 : index
      %dim_20 = tensor.dim %arg0, %c0_19 : tensor<?x?x?xf32>
      %8 = arith.muli %c1_18, %dim_20 : index
      %c1_21 = arith.constant 1 : index
      %dim_22 = tensor.dim %arg0, %c1_21 : tensor<?x?x?xf32>
      %9 = arith.muli %8, %dim_22 : index
      %10 = arith.subi %9, %arg7 : index
      %11 = arith.minsi %arg5, %10 : index
      %12 = arith.divsi %11, %arg6 : index
      %13 = arith.muli %12, %arg6 : index
      %14 = scf.for %arg9 = %c0_8 to %13 step %arg6 iter_args(%arg10 = %arg8) -> (tensor<?xf32>) {
        %17 = arith.addi %arg7, %arg9 : index
        %c0_23 = arith.constant 0 : index
        %c1_24 = arith.constant 1 : index
        %c1_25 = arith.constant 1 : index
        %dim_26 = tensor.dim %collapsed, %c1_25 : tensor<?x?xf32>
        %extracted_slice = tensor.extract_slice %collapsed[%17, %c0_23] [%arg6, %dim_26] [%c1_24, %c1_24] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_27 = arith.constant 0 : index
        %c1_28 = arith.constant 1 : index
        %c1_29 = arith.constant 1 : index
        %dim_30 = tensor.dim %collapsed_4, %c1_29 : tensor<?x?xf32>
        %extracted_slice_31 = tensor.extract_slice %collapsed_4[%17, %c0_27] [%arg6, %dim_30] [%c1_28, %c1_28] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_32 = arith.constant 0 : index
        %c1_33 = arith.constant 1 : index
        %c1_34 = arith.constant 1 : index
        %dim_35 = tensor.dim %collapsed_5, %c1_34 : tensor<?x?xf32>
        %extracted_slice_36 = tensor.extract_slice %collapsed_5[%17, %c0_32] [%arg6, %dim_35] [%c1_33, %c1_33] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_37 = arith.constant 0 : index
        %c1_38 = arith.constant 1 : index
        %c1_39 = arith.constant 1 : index
        %dim_40 = tensor.dim %collapsed_6, %c1_39 : tensor<?x?xf32>
        %extracted_slice_41 = tensor.extract_slice %collapsed_6[%17, %c0_37] [%arg6, %dim_40] [%c1_38, %c1_38] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_42 = arith.constant 0 : index
        %c1_43 = arith.constant 1 : index
        %extracted_slice_44 = tensor.extract_slice %arg10[%17] [%arg6] [%c1_43] : tensor<?xf32> to tensor<?xf32>
        %18 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_31, %extracted_slice_36, %extracted_slice_41 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_44 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_47: f32, %in_48: f32, %in_49: f32, %out: f32):
          %19 = arith.addf %in, %in_47 : f32
          %20 = arith.mulf %19, %in_48 : f32
          %21 = arith.addf %20, %in_49 : f32
          %22 = arith.addf %out, %21 : f32
          linalg.yield %22 : f32
        } -> tensor<?xf32>
        %c0_45 = arith.constant 0 : index
        %c1_46 = arith.constant 1 : index
        %inserted_slice = tensor.insert_slice %18 into %arg10[%17] [%arg6] [%c1_46] : tensor<?xf32> into tensor<?xf32>
        scf.yield %inserted_slice : tensor<?xf32>
      }
      %15 = arith.cmpi slt, %13, %11 : index
      %16 = scf.if %15 -> (tensor<?xf32>) {
        %17 = arith.subi %9, %arg6 : index
        %c0_23 = arith.constant 0 : index
        %c1_24 = arith.constant 1 : index
        %c1_25 = arith.constant 1 : index
        %dim_26 = tensor.dim %collapsed, %c1_25 : tensor<?x?xf32>
        %extracted_slice = tensor.extract_slice %collapsed[%17, %c0_23] [%arg6, %dim_26] [%c1_24, %c1_24] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_27 = arith.constant 0 : index
        %c1_28 = arith.constant 1 : index
        %c1_29 = arith.constant 1 : index
        %dim_30 = tensor.dim %collapsed_4, %c1_29 : tensor<?x?xf32>
        %extracted_slice_31 = tensor.extract_slice %collapsed_4[%17, %c0_27] [%arg6, %dim_30] [%c1_28, %c1_28] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_32 = arith.constant 0 : index
        %c1_33 = arith.constant 1 : index
        %c1_34 = arith.constant 1 : index
        %dim_35 = tensor.dim %collapsed_5, %c1_34 : tensor<?x?xf32>
        %extracted_slice_36 = tensor.extract_slice %collapsed_5[%17, %c0_32] [%arg6, %dim_35] [%c1_33, %c1_33] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_37 = arith.constant 0 : index
        %c1_38 = arith.constant 1 : index
        %c1_39 = arith.constant 1 : index
        %dim_40 = tensor.dim %collapsed_6, %c1_39 : tensor<?x?xf32>
        %extracted_slice_41 = tensor.extract_slice %collapsed_6[%17, %c0_37] [%arg6, %dim_40] [%c1_38, %c1_38] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_42 = arith.constant 0 : index
        %c1_43 = arith.constant 1 : index
        %extracted_slice_44 = tensor.extract_slice %14[%17] [%arg6] [%c1_43] : tensor<?xf32> to tensor<?xf32>
        %18 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_31, %extracted_slice_36, %extracted_slice_41 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_44 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_47: f32, %in_48: f32, %in_49: f32, %out: f32):
          %19 = arith.addf %in, %in_47 : f32
          %20 = arith.mulf %19, %in_48 : f32
          %21 = arith.addf %20, %in_49 : f32
          %22 = arith.addf %out, %21 : f32
          linalg.yield %22 : f32
        } -> tensor<?xf32>
        %c0_45 = arith.constant 0 : index
        %c1_46 = arith.constant 1 : index
        %inserted_slice = tensor.insert_slice %18 into %14[%17] [%arg6] [%c1_46] : tensor<?xf32> into tensor<?xf32>
        scf.yield %inserted_slice : tensor<?xf32>
      } else {
        scf.yield %14 : tensor<?xf32>
      }
      scf.yield %16 : tensor<?xf32>
    } {ascendc.parallel}
    %c0_14 = arith.constant 0 : index
    %dim_15 = tensor.dim %4, %c0_14 : tensor<?x?xf32>
    %c1_16 = arith.constant 1 : index
    %dim_17 = tensor.dim %4, %c1_16 : tensor<?x?xf32>
    %expanded = tensor.expand_shape %7 [[0, 1]] output_shape [%dim_15, %dim_17] : tensor<?xf32> into tensor<?x?xf32>
    return %expanded : tensor<?x?xf32>
  }
}


