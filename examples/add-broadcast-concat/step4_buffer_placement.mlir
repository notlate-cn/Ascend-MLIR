#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_concat(%arg0: memref<?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf16>, %arg3: memref<?x?xf16>, %arg4: i64, %arg5: i64) -> memref<?x?xf16> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg5 : i64 to index
    %1 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?xf16>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf16>
    %2 = arith.muli %dim, %c2 : index
    %alloc = memref.alloc(%2, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %subview = memref.subview %alloc[0, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %3 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %subview) -> (memref<?x?xf16, strided<[?, 1]>>) {
      %5 = affine.min #map(%arg6)[%dim, %1]
      %subview_4 = memref.subview %arg0[%arg6] [%5] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg1[%arg6, 0] [%5, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg7[%arg6, 0] [%5, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %6 = scf.for %arg8 = %c0 to %5 step %0 iter_args(%arg9 = %subview_6) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %7 = affine.min #map(%arg8)[%5, %0]
        %subview_7 = memref.subview %subview_4[%arg8] [%7] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %c0_8 = arith.constant 0 : index
        %dim_9 = memref.dim %subview_7, %c0_8 : memref<?xf16, strided<[1], offset: ?>>
        %alloc_10 = memref.alloc(%dim_9) : memref<?xf16, 9 : i32>
        memref.copy %subview_7, %alloc_10 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, 9 : i32>
        %subview_11 = memref.subview %subview_5[%arg8, 0] [%7, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_12 = memref.subview %arg9[%arg8, 0] [%7, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_13 = arith.constant 0 : index
        %dim_14 = memref.dim %subview_12, %c0_13 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_15 = arith.constant 1 : index
        %dim_16 = memref.dim %subview_12, %c1_15 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_17 = memref.alloc(%dim_14, %dim_16) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = "broadcast_add"} ins(%alloc_10, %subview_11 : memref<?xf16, 9 : i32>, memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%alloc_17 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_18: f16, %out: f16):
          %8 = arith.addf %in, %in_18 : f16
          linalg.yield %8 : f16
        }
        memref.copy %subview_12, %subview_12 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_10 : memref<?xf16, 9 : i32>
        memref.copy %alloc_17, %subview_12 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_17 : memref<?x?xf16, 10 : i32>
        scf.yield %arg9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %6, %subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg7 : memref<?x?xf16, strided<[?, 1]>>
    }
    memref.copy %3, %subview : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1]>>
    %subview_1 = memref.subview %alloc[%dim, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %dim_2 = memref.dim %arg2, %c0 : memref<?xf16>
    %dim_3 = memref.dim %arg3, %c1 : memref<?x?xf16>
    %4 = scf.for %arg6 = %c0 to %dim_2 step %1 iter_args(%arg7 = %subview_1) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
      %5 = affine.min #map(%arg6)[%dim_2, %1]
      %subview_4 = memref.subview %arg2[%arg6] [%5] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg3[%arg6, 0] [%5, %dim_3] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg7[%arg6, 0] [%5, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %6 = scf.for %arg8 = %c0 to %5 step %0 iter_args(%arg9 = %subview_6) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %7 = affine.min #map(%arg8)[%5, %0]
        %subview_7 = memref.subview %subview_4[%arg8] [%7] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %c0_8 = arith.constant 0 : index
        %dim_9 = memref.dim %subview_7, %c0_8 : memref<?xf16, strided<[1], offset: ?>>
        %alloc_10 = memref.alloc(%dim_9) : memref<?xf16, 9 : i32>
        memref.copy %subview_7, %alloc_10 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, 9 : i32>
        %subview_11 = memref.subview %subview_5[%arg8, 0] [%7, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_12 = memref.subview %arg9[%arg8, 0] [%7, %dim_3] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_13 = arith.constant 0 : index
        %dim_14 = memref.dim %subview_12, %c0_13 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_15 = arith.constant 1 : index
        %dim_16 = memref.dim %subview_12, %c1_15 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_17 = memref.alloc(%dim_14, %dim_16) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"], library_call = "broadcast_mul"} ins(%alloc_10, %subview_11 : memref<?xf16, 9 : i32>, memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%alloc_17 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_18: f16, %out: f16):
          %8 = arith.mulf %in, %in_18 : f16
          linalg.yield %8 : f16
        }
        memref.copy %subview_12, %subview_12 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_10 : memref<?xf16, 9 : i32>
        memref.copy %alloc_17, %subview_12 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_17 : memref<?x?xf16, 10 : i32>
        scf.yield %arg9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %6, %subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg7 : memref<?x?xf16, strided<[?, 1], offset: ?>>
    }
    memref.copy %4, %subview_1 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    return %alloc : memref<?x?xf16>
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

