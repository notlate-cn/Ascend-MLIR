#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_concat(%arg0: tensor<?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?xf16>, %arg3: tensor<?x?xf16>, %arg4: i64, %arg5: i64) -> tensor<?x?xf16> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg5 : i64 to index
    %1 = arith.index_cast %arg4 : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
    %2 = arith.muli %dim, %c2 : index
    %3 = tensor.empty(%2, %dim_0) : tensor<?x?xf16>
    %extracted_slice = tensor.extract_slice %3[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %4 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %extracted_slice) -> (tensor<?x?xf16>) {
      %6 = affine.min #map(%arg6)[%dim, %1]
      %extracted_slice_5 = tensor.extract_slice %arg0[%arg6] [%6] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_6 = tensor.extract_slice %arg1[%arg6, 0] [%6, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_7 = tensor.extract_slice %arg7[%arg6, 0] [%6, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %7 = scf.for %arg8 = %c0 to %6 step %0 iter_args(%arg9 = %extracted_slice_7) -> (tensor<?x?xf16>) {
        %8 = affine.min #map(%arg8)[%6, %0]
        %extracted_slice_9 = tensor.extract_slice %extracted_slice_5[%arg8] [%8] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_10 = tensor.extract_slice %extracted_slice_6[%arg8, 0] [%8, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_11 = tensor.extract_slice %arg9[%arg8, 0] [%8, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %9 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = "broadcast_add"} ins(%extracted_slice_9, %extracted_slice_10 : tensor<?xf16>, tensor<?x?xf16>) outs(%extracted_slice_11 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_13: f16, %out: f16):
          %10 = arith.addf %in, %in_13 : f16
          linalg.yield %10 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_12 = tensor.insert_slice %9 into %arg9[%arg8, 0] [%8, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_12 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice_8 = tensor.insert_slice %7 into %arg7[%arg6, 0] [%6, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice_8 : tensor<?x?xf16>
    } {ascendc.parallel = true}
    %inserted_slice = tensor.insert_slice %4 into %3[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
    %extracted_slice_1 = tensor.extract_slice %inserted_slice[%dim, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
    %dim_2 = tensor.dim %arg2, %c0 : tensor<?xf16>
    %dim_3 = tensor.dim %arg3, %c1 : tensor<?x?xf16>
    %5 = scf.for %arg6 = %c0 to %dim_2 step %1 iter_args(%arg7 = %extracted_slice_1) -> (tensor<?x?xf16>) {
      %6 = affine.min #map(%arg6)[%dim_2, %1]
      %extracted_slice_5 = tensor.extract_slice %arg2[%arg6] [%6] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_6 = tensor.extract_slice %arg3[%arg6, 0] [%6, %dim_3] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_7 = tensor.extract_slice %arg7[%arg6, 0] [%6, %dim_3] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %7 = scf.for %arg8 = %c0 to %6 step %0 iter_args(%arg9 = %extracted_slice_7) -> (tensor<?x?xf16>) {
        %8 = affine.min #map(%arg8)[%6, %0]
        %extracted_slice_9 = tensor.extract_slice %extracted_slice_5[%arg8] [%8] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_10 = tensor.extract_slice %extracted_slice_6[%arg8, 0] [%8, %dim_3] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_11 = tensor.extract_slice %arg9[%arg8, 0] [%8, %dim_3] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %9 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = "broadcast_mul"} ins(%extracted_slice_9, %extracted_slice_10 : tensor<?xf16>, tensor<?x?xf16>) outs(%extracted_slice_11 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_13: f16, %out: f16):
          %10 = arith.mulf %in, %in_13 : f16
          linalg.yield %10 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_12 = tensor.insert_slice %9 into %arg9[%arg8, 0] [%8, %dim_3] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_12 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice_8 = tensor.insert_slice %7 into %arg7[%arg6, 0] [%6, %dim_3] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice_8 : tensor<?x?xf16>
    } {ascendc.parallel = true}
    %inserted_slice_4 = tensor.insert_slice %5 into %inserted_slice[%dim, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
    return %inserted_slice_4 : tensor<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_mul"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.param.constant true -> !transform.any_param
    %4 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %5 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %6 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %3 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %6 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %3 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %6 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    transform.yield 
  }
}

