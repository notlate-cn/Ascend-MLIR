#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @broadcast_add_reducesum(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: i64, %arg3: i64) -> memref<?xf16> {
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = arith.index_cast %arg2 : i64 to index
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf16>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?xf16>
    linalg.fill ins(%cst : f16) outs(%alloc : memref<?xf16>)
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf16>
    %2 = scf.for %arg4 = %c0 to %dim step %1 iter_args(%arg5 = %alloc) -> (memref<?xf16>) {
      %3 = affine.min #map(%arg4)[%dim, %1]
      %subview = memref.subview %arg0[%arg4] [%3] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_1 = memref.subview %arg1[%arg4, 0] [%3, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %arg5[%arg4] [%3] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %4 = scf.for %arg6 = %c0 to %3 step %0 iter_args(%arg7 = %subview_2) -> (memref<?xf16, strided<[1], offset: ?>>) {
        %5 = affine.min #map(%arg6)[%3, %0]
        %subview_3 = memref.subview %subview[%arg6] [%5] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_4 = memref.subview %subview_1[%arg6, 0] [%5, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_5 = memref.subview %arg7[%arg6] [%5] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "reduction"]} ins(%subview_3, %subview_4 : memref<?xf16, strided<[1], offset: ?>>, memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_5 : memref<?xf16, strided<[1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector"} {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %6 = arith.addf %in, %in_6 : f16
          %7 = arith.addf %out, %6 : f16
          linalg.yield %7 : f16
        }
        memref.copy %subview_5, %subview_5 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        scf.yield %arg7 : memref<?xf16, strided<[1], offset: ?>>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      memref.copy %4, %subview_2 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      scf.yield %arg5 : memref<?xf16>
    } {ascendc.parallel = true}
    return %2 : memref<?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.param.constant true -> !transform.any_param
    transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    transform.yield 
  }
}

