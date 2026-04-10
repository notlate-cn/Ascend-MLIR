#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_concat(%arg0: tensor<?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?xf16>, %arg3: tensor<?x?xf16>, %arg4: i64, %arg5: i64) -> tensor<?x?xf16> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg5 : i64 to index
    %1 = arith.index_cast %arg4 : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %3 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %2) -> (tensor<?x?xf16>) {
      %7 = affine.min #map(%arg6)[%dim, %1]
      %extracted_slice = tensor.extract_slice %arg0[%arg6] [%7] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_8 = tensor.extract_slice %arg1[%arg6, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_9 = tensor.extract_slice %arg7[%arg6, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %8 = scf.for %arg8 = %c0 to %7 step %0 iter_args(%arg9 = %extracted_slice_9) -> (tensor<?x?xf16>) {
        %9 = affine.min #map(%arg8)[%7, %0]
        %extracted_slice_11 = tensor.extract_slice %extracted_slice[%arg8] [%9] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_12 = tensor.extract_slice %extracted_slice_8[%arg8, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_13 = tensor.extract_slice %arg9[%arg8, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %10 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_11, %extracted_slice_12 : tensor<?xf16>, tensor<?x?xf16>) outs(%extracted_slice_13 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_15: f16, %out: f16):
          %11 = arith.addf %in, %in_15 : f16
          linalg.yield %11 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_14 = tensor.insert_slice %10 into %arg9[%arg8, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_14 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice_10 = tensor.insert_slice %8 into %arg7[%arg6, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice_10 : tensor<?x?xf16>
    } {ascendc.parallel = true}
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?xf16>
    %dim_2 = tensor.dim %arg3, %c1 : tensor<?x?xf16>
    %4 = scf.for %arg6 = %c0 to %dim_1 step %1 iter_args(%arg7 = %2) -> (tensor<?x?xf16>) {
      %7 = affine.min #map(%arg6)[%dim_1, %1]
      %extracted_slice = tensor.extract_slice %arg2[%arg6] [%7] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_8 = tensor.extract_slice %arg3[%arg6, 0] [%7, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_9 = tensor.extract_slice %arg7[%arg6, 0] [%7, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %8 = scf.for %arg8 = %c0 to %7 step %0 iter_args(%arg9 = %extracted_slice_9) -> (tensor<?x?xf16>) {
        %9 = affine.min #map(%arg8)[%7, %0]
        %extracted_slice_11 = tensor.extract_slice %extracted_slice[%arg8] [%9] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_12 = tensor.extract_slice %extracted_slice_8[%arg8, 0] [%9, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_13 = tensor.extract_slice %arg9[%arg8, 0] [%9, %dim_2] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %10 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_11, %extracted_slice_12 : tensor<?xf16>, tensor<?x?xf16>) outs(%extracted_slice_13 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_15: f16, %out: f16):
          %11 = arith.mulf %in, %in_15 : f16
          linalg.yield %11 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_14 = tensor.insert_slice %10 into %arg9[%arg8, 0] [%9, %dim_2] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_14 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice_10 = tensor.insert_slice %8 into %arg7[%arg6, 0] [%7, %dim_2] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice_10 : tensor<?x?xf16>
    } {ascendc.parallel = true}
    %dim_3 = tensor.dim %3, %c0 : tensor<?x?xf16>
    %dim_4 = tensor.dim %3, %c1 : tensor<?x?xf16>
    %dim_5 = tensor.dim %4, %c0 : tensor<?x?xf16>
    %dim_6 = tensor.dim %4, %c1 : tensor<?x?xf16>
    %5 = affine.apply #map3()[%dim_3, %dim_5]
    %6 = tensor.empty(%5, %dim_4) : tensor<?x?xf16>
    %inserted_slice = tensor.insert_slice %3 into %6[0, 0] [%dim_3, %dim_4] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
    %inserted_slice_7 = tensor.insert_slice %4 into %inserted_slice[%dim_3, 0] [%dim_5, %dim_6] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
    return %inserted_slice_7 : tensor<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.tensor.decompose_concat
    } : !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.foreach %1 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      %tiled_linalg_op, %loops = transform.structured.tile_using_for %arg1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
      transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
      transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
      transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    }
    transform.yield 
  }
}

