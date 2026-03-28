#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: i64, %arg6: i64) -> memref<?x?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg6 : i64 to index
    %1 = arith.index_cast %arg5 : i64 to index
    %dim = memref.dim %arg0, %c1 : memref<?x?xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xf16>
    %subview = memref.subview %arg0[0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %subview_1 = memref.subview %arg0[%dim_0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %2 = affine.apply #map()[%dim_0]
    %alloc = memref.alloc(%2, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %subview_2 = memref.subview %alloc[%dim_0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %subview_3 = memref.subview %alloc[0, 0] [%dim_0, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    scf.for %arg7 = %c0 to %dim_0 step %1 {
      %3 = affine.min #map1(%arg7)[%dim_0, %1]
      %subview_4 = memref.subview %subview[%arg7, 0] [%3, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg1[%arg7] [%3] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg3[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %subview_3[%arg7, 0] [%3, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg8 = %c0 to %3 step %0 {
        %4 = affine.min #map1(%arg8)[%3, %0]
        %subview_8 = memref.subview %subview_4[%arg8, 0] [%4, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_5[%arg8] [%4] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %subview_7[%arg8, 0] [%4, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map2, #map3, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_8, %subview_9, %subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>>, memref<?xf16, strided<[1], offset: ?>>, memref<?xf16, strided<[1]>>) outs(%subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_11: f16, %in_12: f16, %out: f16):
          %5 = arith.maximumf %in, %cst : f16
          %6 = arith.addf %5, %in_11 : f16
          %7 = arith.mulf %6, %in_12 : f16
          linalg.yield %7 : f16
        }
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
    } {ascendc.parallel = true}
    scf.for %arg7 = %c0 to %dim_0 step %1 {
      %3 = affine.min #map1(%arg7)[%dim_0, %1]
      %subview_4 = memref.subview %subview_1[%arg7, 0] [%3, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg2[%arg7] [%3] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg4[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %subview_2[%arg7, 0] [%3, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.for %arg8 = %c0 to %3 step %0 {
        %4 = affine.min #map1(%arg8)[%3, %0]
        %subview_8 = memref.subview %subview_4[%arg8, 0] [%4, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_9 = memref.subview %subview_5[%arg8] [%4] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %subview_7[%arg8, 0] [%4, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map2, #map3, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_8, %subview_9, %subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>>, memref<?xf16, strided<[1], offset: ?>>, memref<?xf16, strided<[1]>>) outs(%subview_10 : memref<?x?xf16, strided<[?, 1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_11: f16, %in_12: f16, %out: f16):
          %5 = arith.maximumf %in, %cst : f16
          %6 = arith.addf %5, %in_11 : f16
          %7 = arith.mulf %6, %in_12 : f16
          linalg.yield %7 : f16
        }
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
    } {ascendc.parallel = true}
    return %alloc : memref<?x?xf16>
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

