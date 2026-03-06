#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module {
  func.func @fc_relu(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> tensor<?x?xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg8 : i64 to index
    %1 = arith.index_cast %arg7 : i64 to index
    %2 = arith.index_cast %arg6 : i64 to index
    %3 = arith.index_cast %arg5 : i64 to index
    %4 = arith.index_cast %arg4 : i64 to index
    %dim = tensor.dim %arg3, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg3, %c1 : tensor<?x?xf32>
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %7 = scf.for %arg9 = %c0 to %dim step %4 iter_args(%arg10 = %arg3) -> (tensor<?x?xf32>) {
      %8 = scf.for %arg11 = %c0 to %dim_0 step %3 iter_args(%arg12 = %arg10) -> (tensor<?x?xf32>) {
        %9 = affine.min #map(%arg9)[%dim, %4]
        %10 = affine.min #map(%arg11)[%dim_0, %3]
        %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
        %extracted_slice = tensor.extract_slice %arg0[%arg9, 0] [%9, %dim_1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_2 = tensor.extract_slice %arg1[0, %arg11] [%dim_1, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_3 = tensor.extract_slice %arg12[%arg9, %arg11] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_4 = tensor.extract_slice %arg2[%arg9, %arg11] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_5 = tensor.extract_slice %6[%arg9, %arg11] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %11 = scf.for %arg13 = %c0 to %9 step %2 iter_args(%arg14 = %extracted_slice_3) -> (tensor<?x?xf32>) {
          %12 = scf.for %arg15 = %c0 to %10 step %1 iter_args(%arg16 = %arg14) -> (tensor<?x?xf32>) {
            %13 = affine.min #map(%arg13)[%9, %2]
            %14 = affine.min #map(%arg15)[%10, %1]
            %extracted_slice_6 = tensor.extract_slice %extracted_slice[%arg13, 0] [%13, %dim_1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[0, %arg15] [%dim_1, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_8 = tensor.extract_slice %extracted_slice_3[%arg13, %arg15] [%13, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %15 = scf.for %arg17 = %c0 to %dim_1 step %0 iter_args(%arg18 = %extracted_slice_8) -> (tensor<?x?xf32>) {
              %18 = affine.min #map(%arg17)[%dim_1, %0]
              %extracted_slice_13 = tensor.extract_slice %extracted_slice_6[0, %arg17] [%13, %18] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %extracted_slice_14 = tensor.extract_slice %extracted_slice_7[%arg17, 0] [%18, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %extracted_slice_15 = tensor.extract_slice %arg18[0, 0] [%13, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %19 = linalg.matmul {ascendc.unit = "AiCore.Cube"} ins(%extracted_slice_13, %extracted_slice_14 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %inserted_slice_16 = tensor.insert_slice %19 into %arg18[0, 0] [%13, %14] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %inserted_slice_16 : tensor<?x?xf32>
            } {ascendc.epilogue = "acc:CO1->VECIN", ascendc.prologue = "lhs:A1->A2,rhs:B1->B2"}
            %extracted_slice_9 = tensor.extract_slice %extracted_slice_4[%arg13, %arg15] [%13, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %16 = linalg.elementwise kind=#linalg.elementwise_kind<add> {ascendc.unit = "AiCore.Vector"} ins(%15, %extracted_slice_9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_8 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %extracted_slice_10 = tensor.extract_slice %extracted_slice_5[%arg13, %arg15] [%13, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_11 = tensor.extract_slice %arg16[%arg13, %arg15] [%13, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %17 = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> {ascendc.unit = "AiCore.Vector"} ins(%16, %extracted_slice_10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_11 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %inserted_slice_12 = tensor.insert_slice %17 into %arg16[%arg13, %arg15] [%13, %14] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %inserted_slice_12 : tensor<?x?xf32>
          }
          scf.yield %12 : tensor<?x?xf32>
        }
        %inserted_slice = tensor.insert_slice %11 into %arg12[%arg9, %arg11] [%9, %10] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %inserted_slice : tensor<?x?xf32>
      } {ascendc.epilogue = "result:VECOUT->GM", ascendc.parallel = true, ascendc.prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"}
      scf.yield %8 : tensor<?x?xf32>
    } {ascendc.parallel = true}
    return %7 : tensor<?x?xf32>
  }
}

