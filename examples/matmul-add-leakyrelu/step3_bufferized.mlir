#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> memref<?x?xf32> attributes {ascendc.kernel_kind = "mix"} {
    %cst = arith.constant 1.000000e-03 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg8 : i64 to index
    %1 = arith.index_cast %arg7 : i64 to index
    %2 = arith.index_cast %arg6 : i64 to index
    %3 = arith.index_cast %arg5 : i64 to index
    %4 = arith.index_cast %arg4 : i64 to index
    %dim = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg3, %c1 : memref<?x?xf32>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    %5 = scf.for %arg9 = %c0 to %dim step %4 iter_args(%arg10 = %alloc) -> (memref<?x?xf32>) {
      %6 = scf.for %arg11 = %c0 to %dim_0 step %3 iter_args(%arg12 = %arg10) -> (memref<?x?xf32>) {
        %7 = affine.min #map(%arg9)[%dim, %4]
        %8 = affine.min #map(%arg11)[%dim_0, %3]
        %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf16>
        %subview = memref.subview %arg0[%arg9, 0] [%7, %dim_1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_2 = memref.subview %arg1[0, %arg11] [%dim_1, %8] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_3 = memref.subview %arg3[%arg9, %arg11] [%7, %8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_4 = memref.subview %arg2[%arg11] [%8] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %alloc_5 = memref.alloc(%7, %8) {alignment = 64 : i64} : memref<?x?xf32>
        %subview_6 = memref.subview %arg12[%arg9, %arg11] [%7, %8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %9 = scf.for %arg13 = %c0 to %7 step %2 iter_args(%arg14 = %subview_6) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>) {
          %10 = scf.for %arg15 = %c0 to %8 step %1 iter_args(%arg16 = %arg14) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>) {
            %11 = affine.min #map(%arg13)[%7, %2]
            %12 = affine.min #map(%arg15)[%8, %1]
            %subview_7 = memref.subview %subview[%arg13, 0] [%11, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
            %subview_8 = memref.subview %subview_2[0, %arg15] [%dim_1, %12] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
            %subview_9 = memref.subview %subview_3[%arg13, %arg15] [%11, %12] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %dim_10 = memref.dim %subview_9, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
            %dim_11 = memref.dim %subview_9, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
            %alloc_12 = memref.alloc(%dim_10, %dim_11) {alignment = 64 : i64} : memref<?x?xf32>
            memref.copy %subview_9, %alloc_12 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32>
            %13 = scf.for %arg17 = %c0 to %dim_1 step %0 iter_args(%arg18 = %alloc_12) -> (memref<?x?xf32>) {
              %14 = affine.min #map(%arg17)[%dim_1, %0]
              %subview_16 = memref.subview %subview_7[0, %arg17] [%11, %14] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
              %subview_17 = memref.subview %subview_8[%arg17, 0] [%14, %12] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
              %subview_18 = memref.subview %arg18[0, 0] [%11, %12] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
              linalg.matmul {ascendc.unit = "AiCore.Cube"} ins(%subview_16, %subview_17 : memref<?x?xf16, strided<[?, 1], offset: ?>>, memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_18 : memref<?x?xf32, strided<[?, 1]>>)
              memref.copy %subview_18, %subview_18 : memref<?x?xf32, strided<[?, 1]>> to memref<?x?xf32, strided<[?, 1]>>
              scf.yield %arg18 : memref<?x?xf32>
            } {ascendc.epilogue = "acc:CO1->VECIN", ascendc.prologue = "lhs:A1->A2,rhs:B1->B2"}
            %subview_13 = memref.subview %subview_4[%arg15] [%12] [1] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
            %subview_14 = memref.subview %alloc_5[%arg13, %arg15] [%11, %12] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%13, %subview_13 : memref<?x?xf32>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector"} {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %14 = arith.addf %in, %in_16 : f32
              linalg.yield %14 : f32
            }
            %subview_15 = memref.subview %arg16[%arg13, %arg15] [%11, %12] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_15 : memref<?x?xf32, strided<[?, 1], offset: ?>>) attrs =  {ascendc.unit = "AiCore.Vector"} {
            ^bb0(%in: f32, %out: f32):
              %14 = arith.mulf %in, %cst : f32
              %15 = arith.maximumf %in, %14 : f32
              linalg.yield %15 : f32
            }
            memref.copy %subview_15, %subview_15 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            scf.yield %arg16 : memref<?x?xf32, strided<[?, 1], offset: ?>>
          }
          scf.yield %10 : memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
        memref.copy %9, %subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        scf.yield %arg12 : memref<?x?xf32>
      } {ascendc.epilogue = "result:VECOUT->GM", ascendc.parallel = true, ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"}
      scf.yield %6 : memref<?x?xf32>
    } {ascendc.parallel = true}
    return %5 : memref<?x?xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %transformed, %new_args:5 = transform.func.add_index_args %0, 5 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["linalg.matmul"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["linalg.generic"]} in %transformed : (!transform.any_op) -> !transform.any_op
    %3:2 = transform.split_handle %2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %4 = transform.param.constant true -> !transform.any_param
    %5 = transform.param.constant "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN" -> !transform.any_param
    %6 = transform.param.constant "result:VECOUT->GM" -> !transform.any_param
    %7 = transform.param.constant "lhs:A1->A2,rhs:B1->B2" -> !transform.any_param
    %8 = transform.param.constant "acc:CO1->VECIN" -> !transform.any_param
    %9 = transform.param.constant "AiCore.Cube" -> !transform.any_param
    %10 = transform.param.constant "AiCore.Vector" -> !transform.any_param
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %3#1 tile_sizes [%new_args#0, %new_args#1] : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %3#0 into %loops#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %1 into %loops#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %11:3 = transform.split_handle %fused_op_0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.annotate %loops#0 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops#1 "ascendc.parallel" = %4 : !transform.any_op, !transform.any_param
    transform.annotate %loops#1 "ascendc.prologue" = %5 : !transform.any_op, !transform.any_param
    transform.annotate %loops#1 "ascendc.epilogue" = %6 : !transform.any_op, !transform.any_param
    %tiled_linalg_op_2, %loops_3:2 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [%new_args#2, %new_args#3] : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %fused_op into %loops_3#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %11#0 into %loops_3#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %12:3 = transform.split_handle %fused_op_6 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %12#0 tile_sizes [0, 0, %new_args#4] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %13 = transform.structured.match ops{["linalg.matmul"]} in %loops_9 : (!transform.any_op) -> !transform.any_op
    transform.annotate %13 "ascendc.unit" = %9 : !transform.any_op, !transform.any_param
    transform.annotate %fused_op_4 "ascendc.unit" = %10 : !transform.any_op, !transform.any_param
    transform.annotate %tiled_linalg_op_2 "ascendc.unit" = %10 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.prologue" = %7 : !transform.any_op, !transform.any_param
    transform.annotate %loops_9 "ascendc.epilogue" = %8 : !transform.any_op, !transform.any_param
    transform.yield 
  }
}
