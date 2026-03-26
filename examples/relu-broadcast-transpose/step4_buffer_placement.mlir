#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d1, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  func.func @relu_transpose_broadcast_add(%arg0: memref<?x1xf16>, %arg1: memref<?x?xf16>, %arg2: i64, %arg3: i64) -> memref<?x?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = arith.index_cast %arg2 : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x1xf16>
    %dim_0 = memref.dim %arg1, %c0 : memref<?x?xf16>
    %alloc = memref.alloc(%dim_0, %dim) {alignment = 64 : i64} : memref<?x?xf16>
    %2 = scf.for %arg4 = %c0 to %dim_0 step %1 iter_args(%arg5 = %alloc) -> (memref<?x?xf16>) {
      %3 = affine.min #map(%arg4)[%dim_0, %1]
      %subview = memref.subview %arg0[0, 0] [%dim, 1] [1, 1] : memref<?x1xf16> to memref<?x1xf16, strided<[1, 1]>>
      %subview_1 = memref.subview %arg1[%arg4, 0] [%3, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_2 = memref.subview %arg5[%arg4, 0] [%3, %dim] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %4 = scf.for %arg6 = %c0 to %3 step %0 iter_args(%arg7 = %subview_2) -> (memref<?x?xf16, strided<[?, 1], offset: ?>>) {
        %5 = affine.min #map(%arg6)[%3, %0]
        %subview_3 = memref.subview %subview[0, 0] [%dim, 1] [1, 1] : memref<?x1xf16, strided<[1, 1]>> to memref<?x1xf16, strided<[1, 1]>>
        %c0_4 = arith.constant 0 : index
        %dim_5 = memref.dim %subview_3, %c0_4 : memref<?x1xf16, strided<[1, 1]>>
        %alloc_6 = memref.alloc(%dim_5) : memref<?x1xf16, 9 : i32>
        memref.copy %subview_3, %alloc_6 : memref<?x1xf16, strided<[1, 1]>> to memref<?x1xf16, 9 : i32>
        %subview_7 = memref.subview %subview_1[%arg6, 0] [%5, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_8 = memref.subview %arg7[%arg6, 0] [%5, %dim] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_9 = arith.constant 0 : index
        %dim_10 = memref.dim %subview_8, %c0_9 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1 = arith.constant 1 : index
        %dim_11 = memref.dim %subview_8, %c1 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_12 = memref.alloc(%dim_10, %dim_11) : memref<?x?xf16, 10 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%alloc_6, %subview_7 : memref<?x1xf16, 9 : i32>, memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%alloc_12 : memref<?x?xf16, 10 : i32>) {
        ^bb0(%in: f16, %in_13: f16, %out: f16):
          %6 = arith.maximumf %in, %cst : f16
          %7 = arith.addf %6, %in_13 : f16
          linalg.yield %7 : f16
        }
        memref.copy %subview_8, %subview_8 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_6 : memref<?x1xf16, 9 : i32>
        memref.copy %alloc_12, %subview_8 : memref<?x?xf16, 10 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_12 : memref<?x?xf16, 10 : i32>
        scf.yield %arg7 : memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
      memref.copy %4, %subview_2 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      scf.yield %arg5 : memref<?x?xf16>
    }
    return %2 : memref<?x?xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:2 = transform.func.add_index_args %0, 2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.param.constant true -> !transform.any_param
    %3 = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %4 = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %5 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [%new_args#0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops "ascendc.parallel" = %2 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %loops_1 "ascendc.prologue" = %3 : !transform.any_op, !transform.any_param
    transform.annotate %loops_1 "ascendc.epilogue" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_0 "ascendc.unit" = %5 : !transform.any_op, !transform.any_param
    transform.loop.hoist_loop_invariant_subsets %loops_1 : !transform.any_op
    transform.yield 
  }
}

