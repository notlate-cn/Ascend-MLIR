#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @broadcast_add_reducesum(%arg0: tensor<?xf16>, %arg1: tensor<?x?xf16>, %arg2: i64, %arg3: i64) -> tensor<?xf16> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = arith.index_cast %arg2 : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf16>
    %2 = tensor.empty(%dim) : tensor<?xf16>
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?xf16>) -> tensor<?xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
    %4 = scf.for %arg4 = %c0 to %dim step %1 iter_args(%arg5 = %3) -> (tensor<?xf16>) {
      %5 = affine.min #map(%arg4)[%dim, %1]
      %extracted_slice = tensor.extract_slice %arg0[%arg4] [%5] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, 0] [%5, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %extracted_slice_2 = tensor.extract_slice %arg5[%arg4] [%5] [1] : tensor<?xf16> to tensor<?xf16>
      %6 = scf.for %arg6 = %c0 to %5 step %0 iter_args(%arg7 = %extracted_slice_2) -> (tensor<?xf16>) {
        %7 = affine.min #map(%arg6)[%5, %0]
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[%arg6] [%7] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[%arg6, 0] [%7, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg6] [%7] [1] : tensor<?xf16> to tensor<?xf16>
        %8 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_3, %extracted_slice_4 : tensor<?xf16>, tensor<?x?xf16>) outs(%extracted_slice_5 : tensor<?xf16>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_7: f16, %out: f16):
          %9 = arith.addf %in, %in_7 : f16
          %10 = arith.addf %out, %9 : f16
          linalg.yield %10 : f16
        } -> tensor<?xf16>
        %inserted_slice_6 = tensor.insert_slice %8 into %arg7[%arg6] [%7] [1] : tensor<?xf16> into tensor<?xf16>
        scf.yield %inserted_slice_6 : tensor<?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4] [%5] [1] : tensor<?xf16> into tensor<?xf16>
      scf.yield %inserted_slice : tensor<?xf16>
    } {ascendc.parallel = true}
    return %4 : tensor<?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    transform.print %1 {name = "--------------------------------  \E5\8E\9F\E5\A7\8B\E8\9E\8D\E5\90\88\E5\AD\90\E5\9B\BE --------------------------------"} : !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %loops {name = "-------------------------------- \E9\A6\96\E6\AC\A1\E5\88\87\E5\87\BA\E5\A4\96\E5\B1\82\E5\BE\AA\E7\8E\AFTB --------------------------------"} : !transform.any_op
    transform.print %tiled_linalg_op {name = "-------------------------------- \E9\A6\96\E6\AC\A1\E5\88\87\E5\88\86\E5\90\8E\E7\9A\84\E5\86\85\E5\B1\82\E5\AD\90\E5\9B\BETb --------------------------------"} : !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.print %loops_1 {name = "-------------------------------- \E4\BA\8C\E6\AC\A1\E5\88\87\E5\87\BA\E5\BE\AA\E7\8E\AFTb--------------------------------"} : !transform.any_op
    transform.print %tiled_linalg_op_0 {name = "-------------------------------- \E4\BA\8C\E6\AC\A1\E5\88\87\E5\90\8E\E7\9A\84\E5\AD\90\E5\9B\BE --------------------------------"} : !transform.any_op
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
    transform.yield 
  }
}

