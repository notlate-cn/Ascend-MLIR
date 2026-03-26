#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0)>
module {
  func.func @relu_index_select_add(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi64>, %arg2: tensor<?xf16>, %arg3: i64, %arg4: i64) -> tensor<?x?xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg4 : i64 to index
    %1 = arith.index_cast %arg3 : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg1, %c0 : tensor<?xi64>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %3 = scf.for %arg5 = %c0 to %dim step %1 iter_args(%arg6 = %2) -> (tensor<?x?xf16>) {
      %4 = affine.min #map(%arg5)[%dim, %1]
      %extracted_slice = tensor.extract_slice %arg1[0] [%dim_0] [1] : tensor<?xi64> to tensor<?xi64>
      %extracted_slice_1 = tensor.extract_slice %arg2[0] [%dim_0] [1] : tensor<?xf16> to tensor<?xf16>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg5, 0] [%4, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %5 = scf.for %arg7 = %c0 to %4 step %0 iter_args(%arg8 = %extracted_slice_2) -> (tensor<?x?xf16>) {
        %6 = affine.min #map(%arg7)[%4, %0]
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[0] [%dim_0] [1] : tensor<?xi64> to tensor<?xi64>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0] [%dim_0] [1] : tensor<?xf16> to tensor<?xf16>
        %extracted_slice_5 = tensor.extract_slice %arg8[%arg7, 0] [%6, %dim_0] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %7 = linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_3, %extracted_slice_4 : tensor<?xi64>, tensor<?xf16>) outs(%extracted_slice_5 : tensor<?x?xf16>) attrs =  {ascendc.unit = "AiCore.Vector", gather_dim = 1 : i64} {
        ^bb0(%in: i64, %in_7: f16, %out: f16):
          %8 = linalg.index 0 : index
          %9 = affine.apply #map3(%arg5, %arg7)[%8]
          %10 = arith.index_cast %in : i64 to index
          %extracted = tensor.extract %arg0[%9, %10] : tensor<?x?xf16>
          %11 = arith.maximumf %extracted, %cst : f16
          %12 = arith.addf %11, %in_7 : f16
          linalg.yield %12 : f16
        } -> tensor<?x?xf16>
        %inserted_slice_6 = tensor.insert_slice %7 into %arg8[%arg7, 0] [%6, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
        scf.yield %inserted_slice_6 : tensor<?x?xf16>
      } {ascendc.epilogue = "dst:VECOUT->GM", ascendc.prologue = "src:GM->VECIN"}
      %inserted_slice = tensor.insert_slice %5 into %arg6[%arg5, 0] [%4, %dim_0] [1, 1] : tensor<?x?xf16> into tensor<?x?xf16>
      scf.yield %inserted_slice : tensor<?x?xf16>
    } {ascendc.parallel = true}
    return %3 : tensor<?x?xf16>
  }
}

