#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_transpose(%arg0: tensor<?x?xf16>, %arg1: tensor<?xf16>, %arg2: tensor<?xf16>, %arg3: i64, %arg4: i64) -> tensor<?x?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i64 to index
    %1 = arith.index_cast %arg3 : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %3 = scf.for %arg5 = %c0 to %dim step %1 iter_args(%arg6 = %2) -> (tensor<?x?xf16>) {
      %7 = affine.min #map(%arg5)[%dim, %1]
      %extracted_slice = tensor.extract_slice %arg0[%arg5, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_5 = tensor.extract_slice %arg1[0] [%dim_0] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_6 = tensor.extract_slice %arg6[%arg5, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %8 = scf.for %arg7 = %c0 to %7 step %0 iter_args(%arg8 = %extracted_slice_6) -> (tensor<?x?xf16>) {
        %9 = affine.min #map(%arg7)[%7, %0]
        %extracted_slice_7 = tensor.extract_slice %extracted_slice[%arg7, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_8 = tensor.extract_slice %extracted_slice_5[0] [%dim_0] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_9 = tensor.extract_slice %arg8[%arg7, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"], library_call = "relu_bias_add"} ins(%extracted_slice_7, %extracted_slice_8 : tensor<?x?xf16>, tensor<?xf16>) outs(%extracted_slice_9 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %11 = arith.maximumf %in, %cst : f16
          %12 = arith.addf %11, %in_11 : f16
          linalg.yield %12 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_10 = tensor.insert_slice %10 into %arg8[%arg7, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_10 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice = tensor.insert_slice %8 into %arg6[%arg5, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice : tensor<?x?xf16>
    } {ascendc.parallel = true}
    %4 = tensor.empty(%dim_0, %dim) : tensor<?x?xf16>
    %dim_1 = tensor.dim %3, %c0 : tensor<?x?xf16>
    %dim_2 = tensor.dim %3, %c1 : tensor<?x?xf16>
    %5 = scf.for %arg5 = %c0 to %dim_2 step %1 iter_args(%arg6 = %4) -> (tensor<?x?xf16>) {
      %7 = affine.min #map(%arg5)[%dim_2, %1]
      %extracted_slice = tensor.extract_slice %3[0, %arg5] [%dim_1, %7] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_5 = tensor.extract_slice %arg6[%arg5, 0] [%7, %dim_1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %8 = scf.for %arg7 = %c0 to %7 step %0 iter_args(%arg8 = %extracted_slice_5) -> (tensor<?x?xf16>) {
        %9 = affine.min #map(%arg7)[%7, %0]
        %extracted_slice_6 = tensor.extract_slice %extracted_slice[0, %arg7] [%dim_1, %9] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_7 = tensor.extract_slice %arg8[%arg7, 0] [%9, %dim_1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %10 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"], library_call = "transpose"} ins(%extracted_slice_6 : tensor<?x?xf16>) outs(%extracted_slice_7 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<?x?xf16>
        %inserted_slice_8 = tensor.insert_slice %10 into %arg8[%arg7, 0] [%9, %dim_1] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_8 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice = tensor.insert_slice %8 into %arg6[%arg5, 0] [%7, %dim_1] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice : tensor<?x?xf16>
    } {ascendc.parallel = true}
    %dim_3 = tensor.dim %5, %c0 : tensor<?x?xf16>
    %dim_4 = tensor.dim %5, %c1 : tensor<?x?xf16>
    %6 = scf.for %arg5 = %c0 to %dim_3 step %1 iter_args(%arg6 = %4) -> (tensor<?x?xf16>) {
      %7 = affine.min #map(%arg5)[%dim_3, %1]
      %extracted_slice = tensor.extract_slice %5[%arg5, 0] [%7, %dim_4] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_5 = tensor.extract_slice %arg2[0] [%dim_4] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_6 = tensor.extract_slice %arg6[%arg5, 0] [%7, %dim_4] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %8 = scf.for %arg7 = %c0 to %7 step %0 iter_args(%arg8 = %extracted_slice_6) -> (tensor<?x?xf16>) {
        %9 = affine.min #map(%arg7)[%7, %0]
        %extracted_slice_7 = tensor.extract_slice %extracted_slice[%arg7, 0] [%9, %dim_4] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_8 = tensor.extract_slice %extracted_slice_5[0] [%dim_4] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_9 = tensor.extract_slice %arg8[%arg7, 0] [%9, %dim_4] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"], library_call = "scale_mul"} ins(%extracted_slice_7, %extracted_slice_8 : tensor<?x?xf16>, tensor<?xf16>) outs(%extracted_slice_9 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %11 = arith.mulf %in, %in_11 : f16
          linalg.yield %11 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_10 = tensor.insert_slice %10 into %arg8[%arg7, 0] [%9, %dim_4] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_10 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice = tensor.insert_slice %8 into %arg6[%arg5, 0] [%7, %dim_4] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice : tensor<?x?xf16>
    } {ascendc.parallel = true}
    return %6 : tensor<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "relu_bias_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "transpose"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "scale_mul"} in %transformed : (!transform.any_op) -> !transform.any_op
    %4 = transform.param.constant true -> !transform.any_param
    %5 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %6 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %7 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %3 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_7 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_9 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_8 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_9 : !transform.any_op
    transform.yield 
  }
}

