#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_gather(%arg0: memref<?x?xf16>, %arg1: memref<?xi32>, %arg2: memref<?xf16>, %arg3: i64, %arg4: i64) -> memref<?x?xf16> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i64 to index
    %1 = arith.index_cast %arg3 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xi32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %alloc_1 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %2 = scf.for %arg5 = %c0 to %dim step %1 iter_args(%arg6 = %alloc_1) -> (memref<?x?xf16>) {
      %4 = affine.min #map(%arg5)[%dim, %1]
      %subview = memref.subview %arg1[0] [%dim_0] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
      %subview_4 = memref.subview %arg0[%arg5, 0] [%4, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg6[%arg5, 0] [%4, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %5 = scf.for %arg7 = %c0 to %4 step %0 iter_args(%arg8 = %subview_5) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %6 = affine.min #map(%arg7)[%4, %0]
        %subview_6 = memref.subview %subview[0] [%dim_0] [1] : memref<?xi32, strided<[1]>> to memref<?xi32, strided<[1]>>
        %subview_7 = memref.subview %subview_4[%arg7, 0] [%6, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_8 = memref.subview %arg8[%arg7, 0] [%6, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = "gather_by_index"} ins(%subview_6, %subview_7 : memref<?xi32, strided<[1]>>, memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_8 : memref<?x?xf16, strided<[?, 1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: i32, %in_9: f16, %out: f16):
          linalg.yield %in_9 : f16
        }
        memref.copy %subview_8, %subview_8 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %arg8 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      memref.copy %5, %subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg6 : memref<?x?xf16>
    } {ascendc.parallel = true}
    %dim_2 = memref.dim %2, %c0 : memref<?x?xf16>
    %dim_3 = memref.dim %2, %c1 : memref<?x?xf16>
    %3 = scf.for %arg5 = %c0 to %dim_2 step %1 iter_args(%arg6 = %alloc) -> (memref<?x?xf16>) {
      %4 = affine.min #map(%arg5)[%dim_2, %1]
      %subview = memref.subview %2[%arg5, 0] [%4, %dim_3] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %arg2[%arg5] [%4] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg6[%arg5, 0] [%4, %dim_3] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %5 = scf.for %arg7 = %c0 to %4 step %0 iter_args(%arg8 = %subview_5) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %6 = affine.min #map(%arg7)[%4, %0]
        %subview_6 = memref.subview %subview[%arg7, 0] [%6, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %subview_4[%arg7] [%6] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_8 = memref.subview %arg8[%arg7, 0] [%6, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"], library_call = "broadcast_add_gathered"} ins(%subview_6, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>>, memref<?xf16, strided<[1], offset: ?>>) outs(%subview_8 : memref<?x?xf16, strided<[?, 1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_9: f16, %out: f16):
          %7 = arith.addf %in, %in_9 : f16
          linalg.yield %7 : f16
        }
        memref.copy %subview_8, %subview_8 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %arg8 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      memref.copy %5, %subview_5 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg6 : memref<?x?xf16>
    } {ascendc.parallel = true}
    return %3 : memref<?x?xf16>
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

