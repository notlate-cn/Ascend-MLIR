#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul_add_leakyrelu(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?xf32>, %arg3: memref<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> memref<?x?xf32> {
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
        %c0_2 = arith.constant 0 : index
        %dim_3 = memref.dim %subview, %c0_2 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_4 = arith.constant 1 : index
        %dim_5 = memref.dim %subview, %c1_4 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_6 = memref.alloc(%dim_3, %dim_5) : memref<?x?xf16, 1 : i32>
        memref.copy %subview, %alloc_6 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, 1 : i32>
        %subview_7 = memref.subview %arg1[0, %arg11] [%dim_1, %8] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c0_8 = arith.constant 0 : index
        %dim_9 = memref.dim %subview_7, %c0_8 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %c1_10 = arith.constant 1 : index
        %dim_11 = memref.dim %subview_7, %c1_10 : memref<?x?xf16, strided<[?, 1], offset: ?>>
        %alloc_12 = memref.alloc(%dim_9, %dim_11) : memref<?x?xf16, 3 : i32>
        memref.copy %subview_7, %alloc_12 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, 3 : i32>
        %subview_13 = memref.subview %arg3[%arg9, %arg11] [%7, %8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_14 = memref.subview %arg2[%arg11] [%8] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
        %c0_15 = arith.constant 0 : index
        %dim_16 = memref.dim %subview_14, %c0_15 : memref<?xf32, strided<[1], offset: ?>>
        %alloc_17 = memref.alloc(%dim_16) : memref<?xf32, 9 : i32>
        memref.copy %subview_14, %alloc_17 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, 9 : i32>
        %alloc_18 = memref.alloc(%7, %8) : memref<?x?xf32, 11 : i32>
        %subview_19 = memref.subview %arg12[%arg9, %arg11] [%7, %8] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %9 = scf.for %arg13 = %c0 to %7 step %2 iter_args(%arg14 = %subview_19) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>) {
          %10 = scf.for %arg15 = %c0 to %8 step %1 iter_args(%arg16 = %arg14) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>) {
            %11 = affine.min #map(%arg13)[%7, %2]
            %12 = affine.min #map(%arg15)[%8, %1]
            %subview_20 = memref.subview %subview[%arg13, 0] [%11, %dim_1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
            %subview_21 = memref.subview %subview_7[0, %arg15] [%dim_1, %12] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
            %subview_22 = memref.subview %subview_13[%arg13, %arg15] [%11, %12] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %dim_23 = memref.dim %subview_22, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
            %dim_24 = memref.dim %subview_22, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
            %alloc_25 = memref.alloc(%dim_23, %dim_24) : memref<?x?xf32, 7 : i32>
            %cst_26 = arith.constant 0.000000e+00 : f32
            linalg.fill ins(%cst_26 : f32) outs(%alloc_25 : memref<?x?xf32, 7 : i32>)
            %13 = scf.for %arg17 = %c0 to %dim_1 step %0 iter_args(%arg18 = %alloc_25) -> (memref<?x?xf32, 7 : i32>) {
              %14 = affine.min #map(%arg17)[%dim_1, %0]
              %subview_40 = memref.subview %subview_20[0, %arg17] [%11, %14] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
              %subview_41 = memref.subview %alloc_6[0, %arg17] [%11, %14] [1, 1] : memref<?x?xf16, 1 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>, 1 : i32>
              %c0_42 = arith.constant 0 : index
              %dim_43 = memref.dim %subview_41, %c0_42 : memref<?x?xf16, strided<[?, 1], offset: ?>, 1 : i32>
              %c1_44 = arith.constant 1 : index
              %dim_45 = memref.dim %subview_41, %c1_44 : memref<?x?xf16, strided<[?, 1], offset: ?>, 1 : i32>
              %alloc_46 = memref.alloc(%dim_43, %dim_45) : memref<?x?xf16, 2 : i32>
              memref.copy %subview_41, %alloc_46 : memref<?x?xf16, strided<[?, 1], offset: ?>, 1 : i32> to memref<?x?xf16, 2 : i32>
              %subview_47 = memref.subview %subview_21[%arg17, 0] [%14, %12] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
              %subview_48 = memref.subview %alloc_12[%arg17, 0] [%14, %12] [1, 1] : memref<?x?xf16, 3 : i32> to memref<?x?xf16, strided<[?, 1], offset: ?>, 3 : i32>
              %c0_49 = arith.constant 0 : index
              %dim_50 = memref.dim %subview_48, %c0_49 : memref<?x?xf16, strided<[?, 1], offset: ?>, 3 : i32>
              %c1_51 = arith.constant 1 : index
              %dim_52 = memref.dim %subview_48, %c1_51 : memref<?x?xf16, strided<[?, 1], offset: ?>, 3 : i32>
              %alloc_53 = memref.alloc(%dim_50, %dim_52) : memref<?x?xf16, 4 : i32>
              memref.copy %subview_48, %alloc_53 : memref<?x?xf16, strided<[?, 1], offset: ?>, 3 : i32> to memref<?x?xf16, 4 : i32>
              %subview_54 = memref.subview %arg18[0, 0] [%11, %12] [1, 1] : memref<?x?xf32, 7 : i32> to memref<?x?xf32, strided<[?, 1]>, 7 : i32>
              linalg.matmul ins(%alloc_46, %alloc_53 : memref<?x?xf16, 2 : i32>, memref<?x?xf16, 4 : i32>) outs(%subview_54 : memref<?x?xf32, strided<[?, 1]>, 7 : i32>)
              memref.copy %subview_54, %subview_54 : memref<?x?xf32, strided<[?, 1]>, 7 : i32> to memref<?x?xf32, strided<[?, 1]>, 7 : i32>
              memref.dealloc %alloc_46 : memref<?x?xf16, 2 : i32>
              memref.dealloc %alloc_53 : memref<?x?xf16, 4 : i32>
              scf.yield %arg18 : memref<?x?xf32, 7 : i32>
            }
            %c0_27 = arith.constant 0 : index
            %dim_28 = memref.dim %alloc_25, %c0_27 : memref<?x?xf32, 7 : i32>
            %c1_29 = arith.constant 1 : index
            %dim_30 = memref.dim %alloc_25, %c1_29 : memref<?x?xf32, 7 : i32>
            %alloc_31 = memref.alloc(%dim_28, %dim_30) : memref<?x?xf32, 9 : i32>
            memref.copy %alloc_25, %alloc_31 : memref<?x?xf32, 7 : i32> to memref<?x?xf32, 9 : i32>
            %subview_32 = memref.subview %subview_14[%arg15] [%12] [1] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
            %subview_33 = memref.subview %alloc_18[%arg13, %arg15] [%11, %12] [1, 1] : memref<?x?xf32, 11 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 11 : i32>
            linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%13, %subview_32 : memref<?x?xf32, 7 : i32>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_33 : memref<?x?xf32, strided<[?, 1], offset: ?>, 11 : i32>) {
            ^bb0(%in: f32, %in_40: f32, %out: f32):
              %14 = arith.addf %in, %in_40 : f32
              linalg.yield %14 : f32
            }
            %subview_34 = memref.subview %arg16[%arg13, %arg15] [%11, %12] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %c0_35 = arith.constant 0 : index
            %dim_36 = memref.dim %subview_34, %c0_35 : memref<?x?xf32, strided<[?, 1], offset: ?>>
            %c1_37 = arith.constant 1 : index
            %dim_38 = memref.dim %subview_34, %c1_37 : memref<?x?xf32, strided<[?, 1], offset: ?>>
            %alloc_39 = memref.alloc(%dim_36, %dim_38) : memref<?x?xf32, 10 : i32>
            linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_33 : memref<?x?xf32, strided<[?, 1], offset: ?>, 11 : i32>) outs(%alloc_39 : memref<?x?xf32, 10 : i32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = arith.mulf %in, %cst : f32
              %15 = arith.maximumf %in, %14 : f32
              linalg.yield %15 : f32
            }
            memref.copy %subview_34, %subview_34 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            memref.copy %alloc_39, %subview_34 : memref<?x?xf32, 10 : i32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            memref.dealloc %alloc_39 : memref<?x?xf32, 10 : i32>
            memref.dealloc %alloc_31 : memref<?x?xf32, 9 : i32>
            memref.dealloc %alloc_25 : memref<?x?xf32, 7 : i32>
            scf.yield %arg16 : memref<?x?xf32, strided<[?, 1], offset: ?>>
          }
          scf.yield %10 : memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
        memref.copy %9, %subview_19 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        memref.dealloc %alloc_6 : memref<?x?xf16, 1 : i32>
        memref.dealloc %alloc_12 : memref<?x?xf16, 3 : i32>
        memref.dealloc %alloc_17 : memref<?xf32, 9 : i32>
        scf.yield %arg12 : memref<?x?xf32>
      }
      scf.yield %6 : memref<?x?xf32>
    }
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

