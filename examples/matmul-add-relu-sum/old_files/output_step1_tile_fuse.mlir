#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#map2 = affine_map<()[s0] -> (s0 - 1)>
module {
  func.func @fc_relu(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: i64, %arg5: i64, %arg6: i64) -> tensor<?x?xf32> {
    %0 = arith.index_cast %arg6 : i64 to index
    %1 = arith.index_cast %arg5 : i64 to index
    %2 = arith.index_cast %arg4 : i64 to index
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %4, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %4, %c1 : tensor<?x?xf32>
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %c0_1 = arith.constant 0 : index
    %dim_2 = tensor.dim %4, %c0_1 : tensor<?x?xf32>
    %c1_3 = arith.constant 1 : index
    %dim_4 = tensor.dim %4, %c1_3 : tensor<?x?xf32>
    %c0_5 = arith.constant 0 : index
    %dim_6 = tensor.dim %6, %c0_5 : tensor<?x?xf32>
    %c1_7 = arith.constant 1 : index
    %dim_8 = tensor.dim %6, %c1_7 : tensor<?x?xf32>
    %c0_9 = arith.constant 0 : index
    %dim_10 = tensor.dim %arg3, %c0_9 : tensor<?x?xf32>
    %c1_11 = arith.constant 1 : index
    %dim_12 = tensor.dim %arg3, %c1_11 : tensor<?x?xf32>
    %c0_13 = arith.constant 0 : index
    %c0_14 = arith.constant 0 : index
    %7 = scf.for %arg7 = %c0_13 to %dim_2 step %2 iter_args(%arg8 = %arg3) -> (tensor<?x?xf32>) {
      %8 = scf.for %arg9 = %c0_14 to %dim_4 step %1 iter_args(%arg10 = %arg8) -> (tensor<?x?xf32>) {
        %9 = affine.min #map(%arg7)[%dim_2, %2]
        %10 = affine.min #map(%arg9)[%dim_4, %1]
        %11 = affine.apply #map1(%9)
        %12 = affine.apply #map1(%10)
        %13 = affine.apply #map1(%9)
        %14 = affine.apply #map1(%10)
        %15 = affine.apply #map1(%9)
        %16 = affine.apply #map1(%10)
        %17 = affine.apply #map1(%9)
        %18 = affine.apply #map1(%10)
        %c0_15 = arith.constant 0 : index
        %19 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %dim_16 = tensor.dim %19, %c0_15 : tensor<?x?xf32>
        %c1_17 = arith.constant 1 : index
        %20 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %dim_18 = tensor.dim %20, %c1_17 : tensor<?x?xf32>
        %c0_19 = arith.constant 0 : index
        %dim_20 = tensor.dim %arg2, %c0_19 : tensor<?x?xf32>
        %c1_21 = arith.constant 1 : index
        %dim_22 = tensor.dim %arg2, %c1_21 : tensor<?x?xf32>
        %c0_23 = arith.constant 0 : index
        %dim_24 = tensor.dim %arg10, %c0_23 : tensor<?x?xf32>
        %c1_25 = arith.constant 1 : index
        %dim_26 = tensor.dim %arg10, %c1_25 : tensor<?x?xf32>
        %21 = affine.apply #map1(%9)
        %22 = affine.apply #map1(%10)
        %23 = affine.apply #map1(%9)
        %24 = affine.apply #map1(%10)
        %25 = affine.apply #map1(%9)
        %26 = affine.apply #map1(%10)
        %27 = affine.apply #map1(%9)
        %28 = affine.apply #map1(%10)
        %c0_27 = arith.constant 0 : index
        %dim_28 = tensor.dim %arg0, %c0_27 : tensor<?x?xf32>
        %c1_29 = arith.constant 1 : index
        %dim_30 = tensor.dim %arg0, %c1_29 : tensor<?x?xf32>
        %c0_31 = arith.constant 0 : index
        %dim_32 = tensor.dim %arg1, %c0_31 : tensor<?x?xf32>
        %c1_33 = arith.constant 1 : index
        %dim_34 = tensor.dim %arg1, %c1_33 : tensor<?x?xf32>
        %c0_35 = arith.constant 0 : index
        %dim_36 = tensor.dim %arg10, %c0_35 : tensor<?x?xf32>
        %c1_37 = arith.constant 1 : index
        %dim_38 = tensor.dim %arg10, %c1_37 : tensor<?x?xf32>
        %29 = affine.apply #map1(%9)
        %30 = affine.apply #map1(%10)
        %31 = affine.apply #map2()[%dim_30]
        %32 = affine.apply #map1(%9)
        %33 = affine.apply #map2()[%dim_30]
        %34 = affine.apply #map2()[%dim_30]
        %35 = affine.apply #map1(%10)
        %36 = affine.apply #map1(%9)
        %37 = affine.apply #map1(%10)
        %extracted_slice = tensor.extract_slice %arg0[%arg7, 0] [%9, %dim_30] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_39 = tensor.extract_slice %arg1[0, %arg9] [%dim_30, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_40 = tensor.extract_slice %arg10[%arg7, %arg9] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %c0_41 = arith.constant 0 : index
        %c1_42 = arith.constant 1 : index
        %c0_43 = arith.constant 0 : index
        %c1_44 = arith.constant 1 : index
        %c0_45 = arith.constant 0 : index
        %c1_46 = arith.constant 1 : index
        %c0_47 = arith.constant 0 : index
        %38 = scf.for %arg11 = %c0_47 to %dim_30 step %0 iter_args(%arg12 = %extracted_slice_40) -> (tensor<?x?xf32>) {
          %45 = affine.min #map(%arg11)[%dim_30, %0]
          %46 = affine.apply #map1(%9)
          %47 = affine.apply #map1(%10)
          %48 = affine.apply #map1(%45)
          %49 = affine.apply #map1(%9)
          %50 = affine.apply #map1(%45)
          %51 = affine.apply #map1(%45)
          %52 = affine.apply #map1(%10)
          %53 = affine.apply #map1(%9)
          %54 = affine.apply #map1(%10)
          %extracted_slice_52 = tensor.extract_slice %extracted_slice[0, %arg11] [%9, %45] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_53 = tensor.extract_slice %extracted_slice_39[%arg11, 0] [%45, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_54 = tensor.extract_slice %arg12[0, 0] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %55 = linalg.matmul ins(%extracted_slice_52, %extracted_slice_53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_54 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %56 = affine.apply #map1(%9)
          %57 = affine.apply #map1(%10)
          %58 = affine.apply #map1(%45)
          %59 = affine.apply #map1(%9)
          %60 = affine.apply #map1(%10)
          %inserted_slice_55 = tensor.insert_slice %55 into %arg12[0, 0] [%9, %10] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_55 : tensor<?x?xf32>
        }
        %extracted_slice_48 = tensor.extract_slice %arg2[%arg7, %arg9] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_49 = tensor.extract_slice %arg10[%arg7, %arg9] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %39 = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%38, %extracted_slice_48 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_49 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %extracted_slice_50 = tensor.extract_slice %6[%arg7, %arg9] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_51 = tensor.extract_slice %arg10[%arg7, %arg9] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %40 = linalg.elementwise kind=#linalg.elementwise_kind<max_signed> ins(%39, %extracted_slice_50 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_51 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %41 = affine.apply #map1(%9)
        %42 = affine.apply #map1(%10)
        %43 = affine.apply #map1(%9)
        %44 = affine.apply #map1(%10)
        %inserted_slice = tensor.insert_slice %40 into %arg10[%arg7, %arg9] [%9, %10] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %inserted_slice : tensor<?x?xf32>
      }
      scf.yield %8 : tensor<?x?xf32>
    }
    return %7 : tensor<?x?xf32>
  }
}

