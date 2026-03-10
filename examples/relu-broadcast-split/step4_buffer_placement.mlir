#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: i64, %arg5: i64) -> (memref<?x?xf16>, memref<?x?xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg5 : i64 to index
    %1 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf16>
    %dim_1 = memref.dim %arg2, %c0 : memref<?xf16>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %2 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %alloc) -> (memref<?x?xf16>) {
      %5 = affine.min #map(%arg6)[%dim, %1]
      %subview_5 = memref.subview %arg0[%arg6, 0] [%5, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg1[%arg6] [%5] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_7 = memref.subview %arg7[%arg6, 0] [%5, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %6 = scf.for %arg8 = %c0 to %5 step %0 iter_args(%arg9 = %subview_7) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %7 = affine.min #map(%arg8)[%5, %0]
        %subview_8 = memref.subview %subview_5[%arg8, 0] [%7, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_9 = arith.constant 0 : index
        %dim_10 = memref.dim %subview_8, %c0_9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_11 = arith.constant 1 : index
        %dim_12 = memref.dim %subview_8, %c1_11 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_13 = memref.alloc(%dim_10, %dim_12) : memref<?x?xf16, 9 : i32>
        memref.copy %subview_8, %alloc_13 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, 9 : i32>
        %subview_14 = memref.subview %subview_6[%arg8] [%7] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_15 = memref.subview %arg9[%arg8, 0] [%7, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_16 = arith.constant 0 : index
        %dim_17 = memref.dim %subview_15, %c0_16 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_18 = arith.constant 1 : index
        %dim_19 = memref.dim %subview_15, %c1_18 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_20 = memref.alloc(%dim_17, %dim_19) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"], library_call = "relu_broadcast_add"} ins(%alloc_13, %subview_14 : memref<?x?xf16, 9 : i32>, memref<?xf16, strided<[1], offset: ?>>) outs(%alloc_20 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_21: f16, %out: f16):
          %8 = arith.maximumf %in, %cst : f16
          %9 = arith.addf %8, %in_21 : f16
          linalg.yield %9 : f16
        }
        memref.copy %subview_15, %subview_15 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_13 : memref<?x?xf16, 9 : i32>
        memref.copy %alloc_20, %subview_15 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_20 : memref<?x?xf16, 10 : i32>
        scf.yield %arg9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %6, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg7 : memref<?x?xf16>
    }
    %subview = memref.subview %2[0, 0] [%dim, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %alloc_2 = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<?x?xf16>
    %alloc_3 = memref.alloc(%dim, %dim_1) {alignment = 64 : i64} : memref<?x?xf16>
    %3 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %alloc_3) -> (memref<?x?xf16>) {
      %5 = affine.min #map(%arg6)[%dim, %1]
      %subview_5 = memref.subview %subview[%arg6, 0] [%5, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg2[0] [%dim_1] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %arg7[%arg6, 0] [%5, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %6 = scf.for %arg8 = %c0 to %5 step %0 iter_args(%arg9 = %subview_7) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %7 = affine.min #map(%arg8)[%5, %0]
        %subview_8 = memref.subview %subview_5[%arg8, 0] [%7, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_9 = arith.constant 0 : index
        %dim_10 = memref.dim %subview_8, %c0_9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_11 = arith.constant 1 : index
        %dim_12 = memref.dim %subview_8, %c1_11 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_13 = memref.alloc(%dim_10, %dim_12) : memref<?x?xf16, 9 : i32>
        memref.copy %subview_8, %alloc_13 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, 9 : i32>
        %subview_14 = memref.subview %subview_6[0] [%dim_1] [1] : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1]>>
        %subview_15 = memref.subview %arg9[%arg8, 0] [%7, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_16 = arith.constant 0 : index
        %dim_17 = memref.dim %subview_15, %c0_16 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_18 = arith.constant 1 : index
        %dim_19 = memref.dim %subview_15, %c1_18 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_20 = memref.alloc(%dim_17, %dim_19) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"], library_call = "split_scale0"} ins(%alloc_13, %subview_14 : memref<?x?xf16, 9 : i32>, memref<?xf16, strided<[1]>>) outs(%alloc_20 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_21: f16, %out: f16):
          %8 = arith.mulf %in, %in_21 : f16
          linalg.yield %8 : f16
        }
        memref.copy %subview_15, %subview_15 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_13 : memref<?x?xf16, 9 : i32>
        memref.copy %alloc_20, %subview_15 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_20 : memref<?x?xf16, 10 : i32>
        scf.yield %arg9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %6, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg7 : memref<?x?xf16>
    }
    %subview_4 = memref.subview %2[0, %dim_1] [%dim, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %4 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %alloc_2) -> (memref<?x?xf16>) {
      %5 = affine.min #map(%arg6)[%dim, %1]
      %subview_5 = memref.subview %subview_4[%arg6, 0] [%5, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %arg3[0] [%dim_1] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_7 = memref.subview %arg7[%arg6, 0] [%5, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %6 = scf.for %arg8 = %c0 to %5 step %0 iter_args(%arg9 = %subview_7) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %7 = affine.min #map(%arg8)[%5, %0]
        %subview_8 = memref.subview %subview_5[%arg8, 0] [%7, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_9 = arith.constant 0 : index
        %dim_10 = memref.dim %subview_8, %c0_9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_11 = arith.constant 1 : index
        %dim_12 = memref.dim %subview_8, %c1_11 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_13 = memref.alloc(%dim_10, %dim_12) : memref<?x?xf16, 9 : i32>
        memref.copy %subview_8, %alloc_13 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, 9 : i32>
        %subview_14 = memref.subview %subview_6[0] [%dim_1] [1] : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1]>>
        %subview_15 = memref.subview %arg9[%arg8, 0] [%7, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_16 = arith.constant 0 : index
        %dim_17 = memref.dim %subview_15, %c0_16 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_18 = arith.constant 1 : index
        %dim_19 = memref.dim %subview_15, %c1_18 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_20 = memref.alloc(%dim_17, %dim_19) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"], library_call = "split_scale1"} ins(%alloc_13, %subview_14 : memref<?x?xf16, 9 : i32>, memref<?xf16, strided<[1]>>) outs(%alloc_20 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_21: f16, %out: f16):
          %8 = arith.mulf %in, %in_21 : f16
          linalg.yield %8 : f16
        }
        memref.copy %subview_15, %subview_15 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_13 : memref<?x?xf16, 9 : i32>
        memref.copy %alloc_20, %subview_15 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_20 : memref<?x?xf16, 10 : i32>
        scf.yield %arg9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %6, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg7 : memref<?x?xf16>
    }
    return %3, %4 : memref<?x?xf16>, memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "relu_broadcast_add"} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "split_scale0"} in %transformed : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["linalg.generic"]} attributes {library_call = "split_scale1"} in %transformed : (!transform.any_op) -> !transform.any_op
    %4 = transform.param.constant true -> !transform.any_param
    %5 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %6 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %7 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %2 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %3 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_3 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_5 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_5 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_4 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_5 : !transform.any_op
    %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_7 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_9 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_8 "ascendc.unit" = %7 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_9 : !transform.any_op
    transform.yield 
  }
}

