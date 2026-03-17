#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @ewop_broadcast_split(%arg0: memref<?x?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: i64, %arg5: i64) -> (memref<?x?xf16>, memref<?x?xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg5 : i64 to index
    %1 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf16>
    %dim_0 = memref.dim %arg2, %c0 : memref<?xf16>
    %subview = memref.subview %arg0[0, 0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1]>>
    %subview_1 = memref.subview %arg0[0, %dim_0] [%dim, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %alloc_2 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf16>
    %2 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %alloc_2) -> (memref<?x?xf16>) {
      %4 = affine.min #map(%arg6)[%dim, %1]
      %subview_3 = memref.subview %subview[%arg6, 0] [%4, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1]>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %arg1[%arg6] [%4] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg2[0] [%dim_0] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_6 = memref.subview %arg7[%arg6, 0] [%4, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %5 = scf.for %arg8 = %c0 to %4 step %0 iter_args(%arg9 = %subview_6) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %6 = affine.min #map(%arg8)[%4, %0]
        %subview_7 = memref.subview %subview_3[%arg8, 0] [%6, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_8 = arith.constant 0 : index
        %dim_9 = memref.dim %subview_7, %c0_8 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1 = arith.constant 1 : index
        %dim_10 = memref.dim %subview_7, %c1 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_11 = memref.alloc(%dim_9, %dim_10) : memref<?x?xf16, 9 : i32>
        memref.copy %subview_7, %alloc_11 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, 9 : i32>
        %subview_12 = memref.subview %subview_4[%arg8] [%6] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_13 = memref.subview %subview_5[0] [%dim_0] [1] : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1]>>
        %subview_14 = memref.subview %arg9[%arg8, 0] [%6, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_15 = arith.constant 0 : index
        %dim_16 = memref.dim %subview_14, %c0_15 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_17 = arith.constant 1 : index
        %dim_18 = memref.dim %subview_14, %c1_17 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_19 = memref.alloc(%dim_16, %dim_18) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_11, %subview_12, %subview_13 : memref<?x?xf16, 9 : i32>, memref<?xf16, strided<[1], offset: ?>>, memref<?xf16, strided<[1]>>) outs(%alloc_19 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_20: f16, %in_21: f16, %out: f16):
          %7 = arith.maximumf %in, %cst : f16
          %8 = arith.addf %7, %in_20 : f16
          %9 = arith.mulf %8, %in_21 : f16
          linalg.yield %9 : f16
        }
        memref.copy %subview_14, %subview_14 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_11 : memref<?x?xf16, 9 : i32>
        memref.copy %alloc_19, %subview_14 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_19 : memref<?x?xf16, 10 : i32>
        scf.yield %arg9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %5, %subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg7 : memref<?x?xf16>
    }
    %3 = scf.for %arg6 = %c0 to %dim step %1 iter_args(%arg7 = %alloc) -> (memref<?x?xf16>) {
      %4 = affine.min #map(%arg6)[%dim, %1]
      %subview_3 = memref.subview %subview_1[%arg6, 0] [%4, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %arg1[%arg6] [%4] [1] : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>
      %subview_5 = memref.subview %arg3[0] [%dim_0] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
      %subview_6 = memref.subview %arg7[%arg6, 0] [%4, %dim_0] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %5 = scf.for %arg8 = %c0 to %4 step %0 iter_args(%arg9 = %subview_6) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %6 = affine.min #map(%arg8)[%4, %0]
        %subview_7 = memref.subview %subview_3[%arg8, 0] [%6, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_8 = arith.constant 0 : index
        %dim_9 = memref.dim %subview_7, %c0_8 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1 = arith.constant 1 : index
        %dim_10 = memref.dim %subview_7, %c1 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_11 = memref.alloc(%dim_9, %dim_10) : memref<?x?xf16, 9 : i32>
        memref.copy %subview_7, %alloc_11 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, 9 : i32>
        %subview_12 = memref.subview %subview_4[%arg8] [%6] [1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_13 = memref.subview %subview_5[0] [%dim_0] [1] : memref<?xf16, strided<[1]>> to memref<?xf16, strided<[1]>>
        %subview_14 = memref.subview %arg9[%arg8, 0] [%6, %dim_0] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_15 = arith.constant 0 : index
        %dim_16 = memref.dim %subview_14, %c0_15 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_17 = arith.constant 1 : index
        %dim_18 = memref.dim %subview_14, %c1_17 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_19 = memref.alloc(%dim_16, %dim_18) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_11, %subview_12, %subview_13 : memref<?x?xf16, 9 : i32>, memref<?xf16, strided<[1], offset: ?>>, memref<?xf16, strided<[1]>>) outs(%alloc_19 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_20: f16, %in_21: f16, %out: f16):
          %7 = arith.maximumf %in, %cst : f16
          %8 = arith.addf %7, %in_20 : f16
          %9 = arith.mulf %8, %in_21 : f16
          linalg.yield %9 : f16
        }
        memref.copy %subview_14, %subview_14 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_11 : memref<?x?xf16, 9 : i32>
        memref.copy %alloc_19, %subview_14 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_19 : memref<?x?xf16, 10 : i32>
        scf.yield %arg9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %5, %subview_6 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg7 : memref<?x?xf16>
    }
    return %2, %3 : memref<?x?xf16>, memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
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

