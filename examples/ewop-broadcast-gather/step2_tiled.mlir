#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_gather(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi32>, %arg2: tensor<?xf16>, %arg3: i64, %arg4: i64) -> tensor<?x?xf16> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i64 to index
    %1 = arith.index_cast %arg3 : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?xi32>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %3 = scf.for %arg5 = %c0 to %dim step %1 iter_args(%arg6 = %2) -> (tensor<?x?xf16>) {
      %5 = affine.min #map(%arg5)[%dim, %1]
      %extracted_slice = tensor.extract_slice %arg1[0] [%dim_0] [1] : tensor<?xi32> to tensor<?xi32>
      %extracted_slice_3 = tensor.extract_slice %arg0[%arg5, 0] [%5, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_4 = tensor.extract_slice %arg6[%arg5, 0] [%5, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %6 = scf.for %arg7 = %c0 to %5 step %0 iter_args(%arg8 = %extracted_slice_4) -> (tensor<?x?xf16>) {
        %7 = affine.min #map(%arg7)[%5, %0]
        %extracted_slice_5 = tensor.extract_slice %extracted_slice[0] [%dim_0] [1] : tensor<?xi32> to tensor<?xi32>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[%arg7, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_7 = tensor.extract_slice %arg8[%arg7, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %8 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = "gather_by_index"} ins(%extracted_slice_5, %extracted_slice_6 : tensor<?xi32>, tensor<?x?xf16>) outs(%extracted_slice_7 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: i32, %in_9: f16, %out: f16):
          linalg.yield %in_9 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_8 = tensor.insert_slice %8 into %arg8[%arg7, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_8 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice = tensor.insert_slice %6 into %arg6[%arg5, 0] [%5, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice : tensor<?x?xf16>
    } {ascendc.parallel = true}
    %dim_1 = tensor.dim %3, %c0 : tensor<?x?xf16>
    %dim_2 = tensor.dim %3, %c1 : tensor<?x?xf16>
    %4 = scf.for %arg5 = %c0 to %dim_1 step %1 iter_args(%arg6 = %2) -> (tensor<?x?xf16>) {
      %5 = affine.min #map(%arg5)[%dim_1, %1]
      %extracted_slice = tensor.extract_slice %3[%arg5, 0] [%5, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_3 = tensor.extract_slice %arg2[%arg5] [%5] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_4 = tensor.extract_slice %arg6[%arg5, 0] [%5, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %6 = scf.for %arg7 = %c0 to %5 step %0 iter_args(%arg8 = %extracted_slice_4) -> (tensor<?x?xf16>) {
        %7 = affine.min #map(%arg7)[%5, %0]
        %extracted_slice_5 = tensor.extract_slice %extracted_slice[%arg7, 0] [%7, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[%arg7] [%7] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_7 = tensor.extract_slice %arg8[%arg7, 0] [%7, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %8 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"], library_call = "broadcast_add_gathered"} ins(%extracted_slice_5, %extracted_slice_6 : tensor<?x?xf16>, tensor<?xf16>) outs(%extracted_slice_7 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_9: f16, %out: f16):
          %9 = arith.addf %in, %in_9 : f16
          linalg.yield %9 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_8 = tensor.insert_slice %8 into %arg8[%arg7, 0] [%7, %dim_2] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_8 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice = tensor.insert_slice %6 into %arg6[%arg5, 0] [%5, %dim_2] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice : tensor<?x?xf16>
    } {ascendc.parallel = true}
    return %4 : tensor<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "gather_by_index"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "broadcast_add_gathered"} in %transformed : (!transform.any_op) -> !transform.any_op
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

