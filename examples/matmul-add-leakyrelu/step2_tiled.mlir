#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {transform.with_named_sequence} {
  func.func @matmul_add_leakyrelu(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?xf32>, %arg3: tensor<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> tensor<?x?xf32> {
    %cst = arith.constant 1.000000e-03 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg8 : i64 to index
    %1 = arith.index_cast %arg7 : i64 to index
    %2 = arith.index_cast %arg6 : i64 to index
    %3 = arith.index_cast %arg5 : i64 to index
    %4 = arith.index_cast %arg4 : i64 to index
    %dim = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg3, %c1 : tensor<?x?xf32>
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %6 = scf.for %arg9 = %c0 to %dim step %4 iter_args(%arg10 = %5) -> (tensor<?x?xf32>) {
      %7 = scf.for %arg11 = %c0 to %dim_0 step %3 iter_args(%arg12 = %arg10) -> (tensor<?x?xf32>) {
        %8 = affine.min #map(%arg9)[%dim, %4]
        %9 = affine.min #map(%arg11)[%dim_0, %3]
        %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
        %extracted_slice = tensor.extract_slice %arg0[%arg9, 0] [%8, %dim_1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_2 = tensor.extract_slice %arg1[0, %arg11] [%dim_1, %9] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %extracted_slice_3 = tensor.extract_slice %arg3[%arg9, %arg11] [%8, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_4 = tensor.extract_slice %arg2[%arg11] [%9] [1] : tensor<?xf32> to tensor<?xf32>
        %extracted_slice_5 = tensor.extract_slice %5[%arg9, %arg11] [%8, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_6 = tensor.extract_slice %arg12[%arg9, %arg11] [%8, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %10 = scf.for %arg13 = %c0 to %8 step %2 iter_args(%arg14 = %extracted_slice_6) -> (tensor<?x?xf32>) {
          %11 = scf.for %arg15 = %c0 to %9 step %1 iter_args(%arg16 = %arg14) -> (tensor<?x?xf32>) {
            %12 = affine.min #map(%arg13)[%8, %2]
            %13 = affine.min #map(%arg15)[%9, %1]
            %extracted_slice_7 = tensor.extract_slice %extracted_slice[%arg13, 0] [%12, %dim_1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
            %extracted_slice_8 = tensor.extract_slice %extracted_slice_2[0, %arg15] [%dim_1, %13] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
            %extracted_slice_9 = tensor.extract_slice %extracted_slice_3[%arg13, %arg15] [%12, %13] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %14 = scf.for %arg17 = %c0 to %dim_1 step %0 iter_args(%arg18 = %extracted_slice_9) -> (tensor<?x?xf32>) {
              %17 = affine.min #map(%arg17)[%dim_1, %0]
              %extracted_slice_14 = tensor.extract_slice %extracted_slice_7[0, %arg17] [%12, %17] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
              %extracted_slice_15 = tensor.extract_slice %extracted_slice_8[%arg17, 0] [%17, %13] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
              %extracted_slice_16 = tensor.extract_slice %arg18[0, 0] [%12, %13] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %18 = linalg.matmul {ascendc.unit = "AiCore.Cube"} ins(%extracted_slice_14, %extracted_slice_15 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%extracted_slice_16 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %inserted_slice_17 = tensor.insert_slice %18 into %arg18[0, 0] [%12, %13] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %inserted_slice_17 : tensor<?x?xf32>
            } {ascendc.epilogue = "acc:CO1->VECIN", ascendc.prologue = "lhs:A1->A2,rhs:B1->B2"}
            %extracted_slice_10 = tensor.extract_slice %extracted_slice_4[%arg15] [%13] [1] : tensor<?xf32> to tensor<?xf32>
            %extracted_slice_11 = tensor.extract_slice %extracted_slice_5[%arg13, %arg15] [%12, %13] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %15 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%14, %extracted_slice_10 : tensor<?x?xf32>, tensor<?xf32>) outs(%extracted_slice_11 : tensor<?x?xf32>) attrs =  {ascendc.unit = "AiCore.Vector"} {
            ^bb0(%in: f32, %in_14: f32, %out: f32):
              %17 = arith.addf %in, %in_14 : f32
              linalg.yield %17 : f32
            } -> tensor<?x?xf32>
            %extracted_slice_12 = tensor.extract_slice %arg16[%arg13, %arg15] [%12, %13] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<?x?xf32>) outs(%extracted_slice_12 : tensor<?x?xf32>) attrs =  {ascendc.unit = "AiCore.Vector"} {
            ^bb0(%in: f32, %out: f32):
              %17 = arith.mulf %in, %cst : f32
              %18 = arith.maximumf %in, %17 : f32
              linalg.yield %18 : f32
            } -> tensor<?x?xf32>
            %inserted_slice_13 = tensor.insert_slice %16 into %arg16[%arg13, %arg15] [%12, %13] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %inserted_slice_13 : tensor<?x?xf32>
          }
          scf.yield %11 : tensor<?x?xf32>
        }
        %inserted_slice = tensor.insert_slice %10 into %arg12[%arg9, %arg11] [%8, %9] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %inserted_slice : tensor<?x?xf32>
      } {ascendc.epilogue = "result:VECOUT->GM", ascendc.parallel = true, ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"}
      scf.yield %7 : tensor<?x?xf32>
    } {ascendc.parallel = true}
    return %6 : tensor<?x?xf32>
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

