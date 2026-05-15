// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module attributes {vector_plan.tiling_infos = [{block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", fields = [{abi_index = 0 : i32, arg_index = 4 : i32, axis_size = -1 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"}, {abi_index = 1 : i32, arg_index = 5 : i32, axis_size = -1 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"}], kernel_id = "kernel_group1__v0"}]} {
  func.func private @kernel_group1__v0(%arg0: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg1: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg2: tensor<?x?x?xf32> {afir.symbolic_shape = "s0,s1,s2"}, %arg3: tensor<?x?xf32> {afir.symbolic_shape = "s0,s1"}, %arg4: index {vector_plan.default_tile_size = 128 : i64}, %arg5: index {vector_plan.default_tile_size = 16 : i64}) -> tensor<?x?xf32> attributes {afir.axis_extent_expr = "(arg0_dim0 * arg0_dim1)", afir.axis_extents = ["(s0*s1)", "s2"], afir.block_dim_expr = "ceil((arg0_dim0 * arg0_dim1)/XBLOCK)", afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.alloc_tensor() copy(%arg3) {afir.symbolic_shapes = ["s0,s1"]} : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_1 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
    %collapsed_2 = tensor.collapse_shape %0 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_3 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %1 = arith.muli %dim, %dim_3 : index
    %2 = scf.for %arg6 = %c0 to %1 step %arg4 iter_args(%arg7 = %collapsed_2) -> (tensor<?xf32>) {
      %dim_6 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
      %dim_7 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
      %3 = arith.muli %dim_6, %dim_7 : index
      %4 = arith.subi %3, %arg6 : index
      %5 = arith.minsi %arg4, %4 : index
      %6 = arith.divsi %5, %arg5 : index
      %7 = arith.muli %6, %arg5 : index
      %8 = scf.for %arg8 = %c0 to %7 step %arg5 iter_args(%arg9 = %arg7) -> (tensor<?xf32>) {
        %11 = arith.addi %arg6, %arg8 : index
        %dim_8 = tensor.dim %collapsed, %c1 : tensor<?x?xf32>
        %extracted_slice = tensor.extract_slice %collapsed[%11, 0] [%arg5, %dim_8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %dim_9 = tensor.dim %collapsed_0, %c1 : tensor<?x?xf32>
        %extracted_slice_10 = tensor.extract_slice %collapsed_0[%11, 0] [%arg5, %dim_9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %dim_11 = tensor.dim %collapsed_1, %c1 : tensor<?x?xf32>
        %extracted_slice_12 = tensor.extract_slice %collapsed_1[%11, 0] [%arg5, %dim_11] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_13 = tensor.extract_slice %arg9[%11] [%arg5] [1] : tensor<?xf32> to tensor<?xf32>
        %12 = linalg.generic {indexing_maps = [#map, #map, #map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_10, %extracted_slice_12 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_13 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_14: f32, %in_15: f32, %out: f32):
          %13 = arith.addf %in, %in_14 : f32
          %14 = arith.mulf %13, %in_15 : f32
          %15 = arith.addf %out, %14 : f32
          linalg.yield %15 : f32
        } -> tensor<?xf32>
        %inserted_slice = tensor.insert_slice %12 into %arg9[%11] [%arg5] [1] : tensor<?xf32> into tensor<?xf32>
        scf.yield %inserted_slice : tensor<?xf32>
      }
      %9 = arith.cmpi slt, %7, %5 : index
      %10 = scf.if %9 -> (tensor<?xf32>) {
        %11 = arith.subi %3, %arg5 : index
        %dim_8 = tensor.dim %collapsed, %c1 : tensor<?x?xf32>
        %extracted_slice = tensor.extract_slice %collapsed[%11, 0] [%arg5, %dim_8] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %dim_9 = tensor.dim %collapsed_0, %c1 : tensor<?x?xf32>
        %extracted_slice_10 = tensor.extract_slice %collapsed_0[%11, 0] [%arg5, %dim_9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %dim_11 = tensor.dim %collapsed_1, %c1 : tensor<?x?xf32>
        %extracted_slice_12 = tensor.extract_slice %collapsed_1[%11, 0] [%arg5, %dim_11] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_13 = tensor.extract_slice %8[%11] [%arg5] [1] : tensor<?xf32> to tensor<?xf32>
        %12 = linalg.generic {indexing_maps = [#map, #map, #map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_10, %extracted_slice_12 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_13 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_14: f32, %in_15: f32, %out: f32):
          %13 = arith.addf %in, %in_14 : f32
          %14 = arith.mulf %13, %in_15 : f32
          %15 = arith.addf %out, %14 : f32
          linalg.yield %15 : f32
        } -> tensor<?xf32>
        %inserted_slice = tensor.insert_slice %12 into %8[%11] [%arg5] [1] : tensor<?xf32> into tensor<?xf32>
        scf.yield %inserted_slice : tensor<?xf32>
      } else {
        scf.yield %8 : tensor<?xf32>
      }
      scf.yield %10 : tensor<?xf32>
    } {ascendc.parallel}
    %dim_4 = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_5 = tensor.dim %arg3, %c1 : tensor<?x?xf32>
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [%dim_4, %dim_5] : tensor<?xf32> into tensor<?x?xf32>
    return %expanded : tensor<?x?xf32>
  }
}


