#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module {
  func.func @kernel(%arg0: tensor<?x?x128xf32>) -> tensor<?x?x128xf32> {
    %c128_i64 = arith.constant 128 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c32_i64 = arith.constant 32 : i64
    %c4_i64 = arith.constant 4 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.17677669529663687 : f64
    %cst_2 = arith.constant dense_resource<torch_tensor_128_torch.float32_5> : tensor<128xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_128_torch.float32_4> : tensor<128xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_128_torch.float32_3> : tensor<128xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_128_512_torch.float32> : tensor<128x512xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_512_torch.float32> : tensor<512xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_512_128_torch.float32> : tensor<512x128xf32>
    %cst_8 = arith.constant 1.000000e-05 : f64
    %cst_9 = arith.constant dense_resource<torch_tensor_128_torch.float32_2> : tensor<128xf32>
    %cst_10 = arith.constant dense_resource<torch_tensor_128_torch.float32_1> : tensor<128xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_128_torch.float32> : tensor<128xf32>
    %cst_12 = arith.constant dense_resource<torch_tensor_128_128_torch.float32> : tensor<128x128xf32>
    %cst_13 = arith.constant dense_resource<torch_tensor_384_torch.float32> : tensor<384xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_384_128_torch.float32> : tensor<384x128xf32>
    %c384_i64 = arith.constant 384 : i64
    %cst_15 = arith.constant 1.280000e+02 : f32
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x128xf32>
    %0 = arith.index_cast %dim : index to i64
    %dim_16 = tensor.dim %arg0, %c1 : tensor<?x?x128xf32>
    %1 = arith.index_cast %dim_16 : index to i64
    %2 = tensor.empty(%dim_16, %dim) : tensor<?x?x128xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<?x?x128xf32>) outs(%2 : tensor<?x?x128xf32>) permutation = [1, 0, 2]
    %3 = tensor.empty() : tensor<128x384xf32>
    %transposed_17 = linalg.transpose ins(%cst_14 : tensor<384x128xf32>) outs(%3 : tensor<128x384xf32>) permutation = [1, 0]
    %4 = arith.cmpi sge, %1, %c0_i64 : i64
    cf.assert %4, "negative values not allowed in new dimensions"
    %5 = tensor.empty(%dim_16) : tensor<?x128x384xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_17 : tensor<128x384xf32>) outs(%5 : tensor<?x128x384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x128x384xf32>
    %7 = tensor.empty(%dim_16, %dim) : tensor<?x?x384xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?x?x384xf32>) -> tensor<?x?x384xf32>
    %9 = linalg.batch_matmul ins(%transposed, %6 : tensor<?x?x128xf32>, tensor<?x128x384xf32>) outs(%8 : tensor<?x?x384xf32>) -> tensor<?x?x384xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %cst_13 : tensor<?x?x384xf32>, tensor<384xf32>) outs(%7 : tensor<?x?x384xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x384xf32>
    %11 = arith.cmpi slt, %1, %c0_i64 : i64
    %12 = arith.select %11, %c1_i64, %1 : i64
    %13 = arith.extui %11 : i1 to i64
    %14 = arith.muli %12, %0 : i64
    %15 = arith.addi %13, %c1_i64 : i64
    %16 = arith.cmpi slt, %0, %c0_i64 : i64
    %17 = arith.select %16, %12, %14 : i64
    %18 = arith.select %16, %15, %13 : i64
    %19 = arith.muli %17, %c384_i64 : i64
    %20 = arith.cmpi sle, %18, %c1_i64 : i64
    cf.assert %20, "must have at most one inferred (negative) dimension"
    %21 = arith.muli %1, %0 : i64
    %22 = arith.muli %21, %c384_i64 : i64
    %23 = arith.divsi %22, %19 : i64
    %24 = arith.select %11, %23, %1 : i64
    %25 = arith.select %16, %23, %0 : i64
    %from_elements = tensor.from_elements %24, %25, %c3_i64, %c128_i64 : tensor<4xi64>
    %reshape = tensor.reshape %10(%from_elements) : (tensor<?x?x384xf32>, tensor<4xi64>) -> tensor<?x?x3x128xf32>
    %26 = arith.index_cast %24 : i64 to index
    %27 = arith.index_cast %25 : i64 to index
    %expanded = tensor.expand_shape %reshape [[0, 1], [2], [3], [4]] output_shape [1, %26, %27, 3, 128] : tensor<?x?x3x128xf32> into tensor<1x?x?x3x128xf32>
    %28 = tensor.empty(%26, %27) : tensor<3x?x?x1x128xf32>
    %transposed_18 = linalg.transpose ins(%expanded : tensor<1x?x?x3x128xf32>) outs(%28 : tensor<3x?x?x1x128xf32>) permutation = [3, 1, 2, 0, 4]
    %collapsed = tensor.collapse_shape %transposed_18 [[0], [1], [2, 3], [4]] : tensor<3x?x?x1x128xf32> into tensor<3x?x?x128xf32>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0, 0] [1, %26, %27, 128] [1, 1, 1, 1] : tensor<3x?x?x128xf32> to tensor<1x?x?x128xf32>
    %collapsed_19 = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x?x?x128xf32> into tensor<?x?x128xf32>
    %extracted_slice_20 = tensor.extract_slice %collapsed[1, 0, 0, 0] [1, %26, %27, 128] [1, 1, 1, 1] : tensor<3x?x?x128xf32> to tensor<1x?x?x128xf32>
    %collapsed_21 = tensor.collapse_shape %extracted_slice_20 [[0, 1], [2], [3]] : tensor<1x?x?x128xf32> into tensor<?x?x128xf32>
    %extracted_slice_22 = tensor.extract_slice %collapsed[2, 0, 0, 0] [1, %26, %27, 128] [1, 1, 1, 1] : tensor<3x?x?x128xf32> to tensor<1x?x?x128xf32>
    %collapsed_23 = tensor.collapse_shape %extracted_slice_22 [[0, 1], [2], [3]] : tensor<1x?x?x128xf32> into tensor<?x?x128xf32>
    %29 = arith.muli %0, %c4_i64 : i64
    %30 = arith.muli %12, %29 : i64
    %31 = arith.cmpi slt, %29, %c0_i64 : i64
    %32 = arith.select %31, %12, %30 : i64
    %33 = arith.select %31, %15, %13 : i64
    %34 = arith.muli %32, %c32_i64 : i64
    %35 = arith.cmpi sle, %33, %c1_i64 : i64
    cf.assert %35, "must have at most one inferred (negative) dimension"
    %36 = arith.muli %24, %25 : i64
    %37 = arith.muli %36, %c128_i64 : i64
    %38 = arith.divsi %37, %34 : i64
    %39 = arith.select %11, %38, %1 : i64
    %40 = arith.select %31, %38, %29 : i64
    %from_elements_24 = tensor.from_elements %39, %40, %c32_i64 : tensor<3xi64>
    %reshape_25 = tensor.reshape %collapsed_19(%from_elements_24) : (tensor<?x?x128xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
    %41 = arith.index_cast %39 : i64 to index
    %42 = arith.index_cast %40 : i64 to index
    %43 = tensor.empty(%42, %41) : tensor<?x?x32xf32>
    %transposed_26 = linalg.transpose ins(%reshape_25 : tensor<?x?x32xf32>) outs(%43 : tensor<?x?x32xf32>) permutation = [1, 0, 2]
    cf.assert %35, "must have at most one inferred (negative) dimension"
    %reshape_27 = tensor.reshape %collapsed_21(%from_elements_24) : (tensor<?x?x128xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
    %transposed_28 = linalg.transpose ins(%reshape_27 : tensor<?x?x32xf32>) outs(%43 : tensor<?x?x32xf32>) permutation = [1, 0, 2]
    cf.assert %35, "must have at most one inferred (negative) dimension"
    %reshape_29 = tensor.reshape %collapsed_23(%from_elements_24) : (tensor<?x?x128xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
    %transposed_30 = linalg.transpose ins(%reshape_29 : tensor<?x?x32xf32>) outs(%43 : tensor<?x?x32xf32>) permutation = [1, 0, 2]
    %44 = arith.select %16, %c1_i64, %0 : i64
    %45 = arith.extui %16 : i1 to i64
    %46 = arith.muli %44, %c4_i64 : i64
    %47 = arith.muli %46, %1 : i64
    %48 = arith.addi %45, %c1_i64 : i64
    %49 = arith.select %11, %46, %47 : i64
    %50 = arith.select %11, %48, %45 : i64
    %51 = arith.muli %49, %c32_i64 : i64
    %52 = arith.cmpi sle, %50, %c1_i64 : i64
    cf.assert %52, "must have at most one inferred (negative) dimension"
    %53 = arith.muli %40, %39 : i64
    %54 = arith.muli %53, %c32_i64 : i64
    %55 = arith.divsi %54, %51 : i64
    %56 = arith.select %16, %55, %0 : i64
    %57 = arith.select %11, %55, %1 : i64
    %from_elements_31 = tensor.from_elements %56, %c4_i64, %57, %c32_i64 : tensor<4xi64>
    %reshape_32 = tensor.reshape %transposed_26(%from_elements_31) : (tensor<?x?x32xf32>, tensor<4xi64>) -> tensor<?x4x?x32xf32>
    cf.assert %52, "must have at most one inferred (negative) dimension"
    %reshape_33 = tensor.reshape %transposed_28(%from_elements_31) : (tensor<?x?x32xf32>, tensor<4xi64>) -> tensor<?x4x?x32xf32>
    cf.assert %52, "must have at most one inferred (negative) dimension"
    %reshape_34 = tensor.reshape %transposed_30(%from_elements_31) : (tensor<?x?x32xf32>, tensor<4xi64>) -> tensor<?x4x?x32xf32>
    %58 = arith.index_cast %56 : i64 to index
    %59 = arith.index_cast %57 : i64 to index
    %60 = tensor.empty(%58, %59) : tensor<?x4x32x?xf32>
    %transposed_35 = linalg.transpose ins(%reshape_33 : tensor<?x4x?x32xf32>) outs(%60 : tensor<?x4x32x?xf32>) permutation = [0, 1, 3, 2]
    %61 = arith.cmpi slt, %56, %c0_i64 : i64
    %62 = arith.select %61, %c1_i64, %56 : i64
    %63 = arith.extui %61 : i1 to i64
    %64 = arith.muli %62, %c128_i64 : i64
    %65 = arith.muli %64, %57 : i64
    %66 = arith.addi %63, %c1_i64 : i64
    %67 = arith.cmpi slt, %57, %c0_i64 : i64
    %68 = arith.select %67, %64, %65 : i64
    %69 = arith.select %67, %66, %63 : i64
    %70 = arith.cmpi sle, %69, %c1_i64 : i64
    cf.assert %70, "must have at most one inferred (negative) dimension"
    %71 = arith.muli %56, %c128_i64 : i64
    %72 = arith.muli %71, %57 : i64
    %73 = arith.divsi %72, %68 : i64
    %74 = arith.select %61, %73, %56 : i64
    %75 = arith.select %67, %73, %57 : i64
    %from_elements_36 = tensor.from_elements %74, %c4_i64, %c32_i64, %75 : tensor<4xi64>
    %reshape_37 = tensor.reshape %transposed_35(%from_elements_36) : (tensor<?x4x32x?xf32>, tensor<4xi64>) -> tensor<?x4x32x?xf32>
    %76 = arith.index_cast %74 : i64 to index
    %77 = arith.maxui %58, %76 : index
    %78 = arith.index_cast %75 : i64 to index
    %collapsed_38 = tensor.collapse_shape %reshape_32 [[0, 1], [2], [3]] : tensor<?x4x?x32xf32> into tensor<?x?x32xf32>
    %collapsed_39 = tensor.collapse_shape %reshape_37 [[0, 1], [2], [3]] : tensor<?x4x32x?xf32> into tensor<?x32x?xf32>
    %79 = arith.muli %77, %c4 : index
    %80 = tensor.empty(%79, %59, %78) : tensor<?x?x?xf32>
    %81 = linalg.fill ins(%cst : f32) outs(%80 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %82 = linalg.batch_matmul ins(%collapsed_38, %collapsed_39 : tensor<?x?x32xf32>, tensor<?x32x?xf32>) outs(%81 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %83 = arith.divsi %79, %c4 : index
    %expanded_40 = tensor.expand_shape %82 [[0, 1], [2], [3]] output_shape [%83, 4, %59, %78] : tensor<?x?x?xf32> into tensor<?x4x?x?xf32>
    %84 = arith.muli %62, %c4_i64 : i64
    %85 = arith.muli %84, %57 : i64
    %86 = arith.select %67, %84, %85 : i64
    %87 = arith.muli %86, %57 : i64
    %88 = arith.addi %69, %c1_i64 : i64
    %89 = arith.select %67, %84, %87 : i64
    %90 = arith.select %67, %88, %63 : i64
    %91 = arith.cmpi sle, %90, %c1_i64 : i64
    cf.assert %91, "must have at most one inferred (negative) dimension"
    %92 = arith.index_cast %83 : index to i64
    %93 = arith.muli %92, %c4_i64 : i64
    %94 = arith.muli %93, %57 : i64
    %95 = arith.muli %94, %75 : i64
    %96 = arith.divsi %95, %89 : i64
    %97 = arith.select %61, %96, %56 : i64
    %98 = arith.select %67, %96, %57 : i64
    %from_elements_41 = tensor.from_elements %97, %c4_i64, %98, %98 : tensor<4xi64>
    %reshape_42 = tensor.reshape %expanded_40(%from_elements_41) : (tensor<?x4x?x?xf32>, tensor<4xi64>) -> tensor<?x4x?x?xf32>
    %99 = arith.index_cast %97 : i64 to index
    %100 = arith.index_cast %98 : i64 to index
    %101 = tensor.empty(%99, %100, %100) : tensor<?x4x?x?xf32>
    %102 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_42 : tensor<?x4x?x?xf32>) outs(%101 : tensor<?x4x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.truncf %cst_1 : f64 to f32
      %192 = arith.mulf %in, %191 : f32
      linalg.yield %192 : f32
    } -> tensor<?x4x?x?xf32>
    %103 = tensor.empty(%99, %100) : tensor<?x4x?xi64>
    %104 = linalg.fill ins(%c0_i64 : i64) outs(%103 : tensor<?x4x?xi64>) -> tensor<?x4x?xi64>
    %105 = tensor.empty(%99, %100) : tensor<?x4x?xf32>
    %106 = linalg.fill ins(%cst_0 : f32) outs(%105 : tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
    %107:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%102 : tensor<?x4x?x?xf32>) outs(%106, %104 : tensor<?x4x?xf32>, tensor<?x4x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_56: i64):
      %191 = linalg.index 3 : index
      %192 = arith.index_cast %191 : index to i64
      %193 = arith.maximumf %in, %out : f32
      %194 = arith.cmpf ogt, %in, %out : f32
      %195 = arith.select %194, %192, %out_56 : i64
      linalg.yield %193, %195 : f32, i64
    } -> (tensor<?x4x?xf32>, tensor<?x4x?xi64>)
    %expanded_43 = tensor.expand_shape %107#0 [[0], [1], [2, 3]] output_shape [%99, 4, %100, 1] : tensor<?x4x?xf32> into tensor<?x4x?x1xf32>
    %108 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %expanded_43 : tensor<?x4x?x?xf32>, tensor<?x4x?x1xf32>) outs(%101 : tensor<?x4x?x?xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.subf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x4x?x?xf32>
    %109 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108 : tensor<?x4x?x?xf32>) outs(%101 : tensor<?x4x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = math.exp %in : f32
      linalg.yield %191 : f32
    } -> tensor<?x4x?x?xf32>
    %110 = tensor.empty(%99, %100) : tensor<?x4x?x1xf32>
    %111 = linalg.fill ins(%cst : f32) outs(%110 : tensor<?x4x?x1xf32>) -> tensor<?x4x?x1xf32>
    %112 = linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%109 : tensor<?x4x?x?xf32>) outs(%111 : tensor<?x4x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.addf %in, %out : f32
      linalg.yield %191 : f32
    } -> tensor<?x4x?x1xf32>
    %113 = linalg.generic {indexing_maps = [#map3, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %112 : tensor<?x4x?x?xf32>, tensor<?x4x?x1xf32>) outs(%101 : tensor<?x4x?x?xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.divf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x4x?x?xf32>
    %114 = arith.maxui %99, %58 : index
    %115 = arith.cmpi eq, %98, %57 : i64
    cf.assert %115, "mismatching contracting dimension"
    %collapsed_44 = tensor.collapse_shape %113 [[0, 1], [2], [3]] : tensor<?x4x?x?xf32> into tensor<?x?x?xf32>
    %collapsed_45 = tensor.collapse_shape %reshape_34 [[0, 1], [2], [3]] : tensor<?x4x?x32xf32> into tensor<?x?x32xf32>
    %116 = arith.muli %114, %c4 : index
    %117 = tensor.empty(%116, %100) : tensor<?x?x32xf32>
    %118 = linalg.fill ins(%cst : f32) outs(%117 : tensor<?x?x32xf32>) -> tensor<?x?x32xf32>
    %119 = linalg.batch_matmul ins(%collapsed_44, %collapsed_45 : tensor<?x?x?xf32>, tensor<?x?x32xf32>) outs(%118 : tensor<?x?x32xf32>) -> tensor<?x?x32xf32>
    %120 = arith.divsi %116, %c4 : index
    %expanded_46 = tensor.expand_shape %119 [[0, 1], [2], [3]] output_shape [%120, 4, %100, 32] : tensor<?x?x32xf32> into tensor<?x4x?x32xf32>
    %121 = tensor.empty(%100) : tensor<?x1x4x32xf32>
    %cast = tensor.cast %expanded_46 : tensor<?x4x?x32xf32> to tensor<1x4x?x32xf32>
    %transposed_47 = linalg.transpose ins(%cast : tensor<1x4x?x32xf32>) outs(%121 : tensor<?x1x4x32xf32>) permutation = [2, 0, 1, 3]
    %collapsed_48 = tensor.collapse_shape %transposed_47 [[0], [1, 2, 3]] : tensor<?x1x4x32xf32> into tensor<?x128xf32>
    %122 = tensor.empty() : tensor<128x128xf32>
    %transposed_49 = linalg.transpose ins(%cst_12 : tensor<128x128xf32>) outs(%122 : tensor<128x128xf32>) permutation = [1, 0]
    %dim_50 = tensor.dim %expanded_46, %c2 : tensor<?x4x?x32xf32>
    %123 = tensor.empty(%dim_50) : tensor<?x128xf32>
    %124 = linalg.fill ins(%cst : f32) outs(%123 : tensor<?x128xf32>) -> tensor<?x128xf32>
    %125 = linalg.matmul ins(%collapsed_48, %transposed_49 : tensor<?x128xf32>, tensor<128x128xf32>) outs(%124 : tensor<?x128xf32>) -> tensor<?x128xf32>
    %126 = linalg.generic {indexing_maps = [#map6, #map7, #map6], iterator_types = ["parallel", "parallel"]} ins(%125, %cst_11 : tensor<?x128xf32>, tensor<128xf32>) outs(%123 : tensor<?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x128xf32>
    %127 = arith.muli %17, %c128_i64 : i64
    cf.assert %20, "must have at most one inferred (negative) dimension"
    %128 = arith.index_cast %dim_50 : index to i64
    %129 = arith.muli %128, %c128_i64 : i64
    %130 = arith.divsi %129, %127 : i64
    %131 = arith.select %11, %130, %1 : i64
    %132 = arith.select %16, %130, %0 : i64
    %from_elements_51 = tensor.from_elements %131, %132, %c128_i64 : tensor<3xi64>
    %reshape_52 = tensor.reshape %126(%from_elements_51) : (tensor<?x128xf32>, tensor<3xi64>) -> tensor<?x?x128xf32>
    %133 = arith.index_cast %131 : i64 to index
    %134 = arith.index_cast %132 : i64 to index
    %135 = tensor.empty(%134, %133) : tensor<?x?x128xf32>
    %transposed_53 = linalg.transpose ins(%reshape_52 : tensor<?x?x128xf32>) outs(%135 : tensor<?x?x128xf32>) permutation = [1, 0, 2]
    %136 = arith.cmpi eq, %dim, %134 : index
    cf.assert %136, "mismatched size for broadcast"
    %137 = arith.cmpi eq, %dim_16, %133 : index
    cf.assert %137, "mismatched size for broadcast"
    %138 = tensor.empty(%dim, %dim_16) : tensor<?x?x128xf32>
    %139 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %transposed_53 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %140 = tensor.empty(%dim, %dim_16) : tensor<?x?x1xf32>
    %141 = linalg.fill ins(%cst : f32) outs(%140 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %142 = linalg.generic {indexing_maps = [#map1, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%139 : tensor<?x?x128xf32>) outs(%141 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.addf %in, %out : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    %143 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.divf %in, %cst_15 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    %144 = arith.index_cast %0 : i64 to index
    %145 = arith.cmpi sge, %0, %c0_i64 : i64
    cf.assert %145, "unimplemented: dynamic negative broadcast sizes"
    %146 = arith.index_cast %1 : i64 to index
    %147 = arith.cmpi sge, %1, %c0_i64 : i64
    cf.assert %147, "unimplemented: dynamic negative broadcast sizes"
    %148 = tensor.empty(%144, %146) : tensor<?x?x128xf32>
    %149 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%148 : tensor<?x?x128xf32>) {
    ^bb0(%out: f32):
      %191 = linalg.index 0 : index
      %192 = linalg.index 1 : index
      %193 = arith.cmpi eq, %dim, %c1 : index
      %194 = arith.select %193, %c0, %191 : index
      %195 = arith.cmpi eq, %dim_16, %c1 : index
      %196 = arith.select %195, %c0, %192 : index
      %extracted = tensor.extract %143[%194, %196, %c0] : tensor<?x?x1xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x?x128xf32>
    %150 = arith.cmpi eq, %dim, %144 : index
    cf.assert %150, "mismatched size for broadcast"
    %151 = arith.cmpi eq, %dim_16, %146 : index
    cf.assert %151, "mismatched size for broadcast"
    %152 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%139, %149 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.subf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %153 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%152, %152 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.mulf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %154 = linalg.generic {indexing_maps = [#map1, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%153 : tensor<?x?x128xf32>) outs(%141 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.addf %in, %out : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    %155 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%154 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.divf %in, %cst_15 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    %156 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%155 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.truncf %cst_8 : f64 to f32
      %192 = arith.addf %in, %191 : f32
      linalg.yield %192 : f32
    } -> tensor<?x?x1xf32>
    %157 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%156 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = math.rsqrt %in : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    cf.assert %145, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %147, "unimplemented: dynamic negative broadcast sizes"
    %158 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%148 : tensor<?x?x128xf32>) {
    ^bb0(%out: f32):
      %191 = linalg.index 0 : index
      %192 = linalg.index 1 : index
      %193 = arith.cmpi eq, %dim, %c1 : index
      %194 = arith.select %193, %c0, %191 : index
      %195 = arith.cmpi eq, %dim_16, %c1 : index
      %196 = arith.select %195, %c0, %192 : index
      %extracted = tensor.extract %157[%194, %196, %c0] : tensor<?x?x1xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x?x128xf32>
    cf.assert %150, "mismatched size for broadcast"
    cf.assert %151, "mismatched size for broadcast"
    %159 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%152, %158 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.mulf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %160 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%159, %cst_10 : tensor<?x?x128xf32>, tensor<128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.mulf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %161 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160, %cst_9 : tensor<?x?x128xf32>, tensor<128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %162 = tensor.empty() : tensor<128x512xf32>
    %transposed_54 = linalg.transpose ins(%cst_7 : tensor<512x128xf32>) outs(%162 : tensor<128x512xf32>) permutation = [1, 0]
    %163 = arith.cmpi sge, %0, %c0_i64 : i64
    cf.assert %163, "negative values not allowed in new dimensions"
    %164 = tensor.empty(%dim) : tensor<?x128x512xf32>
    %165 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_54 : tensor<128x512xf32>) outs(%164 : tensor<?x128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x128x512xf32>
    %166 = tensor.empty(%dim, %dim_16) : tensor<?x?x512xf32>
    %167 = linalg.fill ins(%cst : f32) outs(%166 : tensor<?x?x512xf32>) -> tensor<?x?x512xf32>
    %168 = linalg.batch_matmul ins(%161, %165 : tensor<?x?x128xf32>, tensor<?x128x512xf32>) outs(%167 : tensor<?x?x512xf32>) -> tensor<?x?x512xf32>
    %169 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%168, %cst_6 : tensor<?x?x512xf32>, tensor<512xf32>) outs(%166 : tensor<?x?x512xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x512xf32>
    %170 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169 : tensor<?x?x512xf32>) outs(%166 : tensor<?x?x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.cmpf ugt, %in, %cst : f32
      %192 = arith.select %191, %in, %cst : f32
      linalg.yield %192 : f32
    } -> tensor<?x?x512xf32>
    %171 = tensor.empty() : tensor<512x128xf32>
    %transposed_55 = linalg.transpose ins(%cst_5 : tensor<128x512xf32>) outs(%171 : tensor<512x128xf32>) permutation = [1, 0]
    cf.assert %163, "negative values not allowed in new dimensions"
    %172 = tensor.empty(%dim) : tensor<?x512x128xf32>
    %173 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_55 : tensor<512x128xf32>) outs(%172 : tensor<?x512x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x512x128xf32>
    %174 = linalg.fill ins(%cst : f32) outs(%138 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
    %175 = linalg.batch_matmul ins(%170, %173 : tensor<?x?x512xf32>, tensor<?x512x128xf32>) outs(%174 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
    %176 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%175, %cst_4 : tensor<?x?x128xf32>, tensor<128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %177 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161, %176 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %178 = linalg.generic {indexing_maps = [#map1, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%177 : tensor<?x?x128xf32>) outs(%141 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.addf %in, %out : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    %179 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%178 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.divf %in, %cst_15 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    cf.assert %145, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %147, "unimplemented: dynamic negative broadcast sizes"
    %180 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%148 : tensor<?x?x128xf32>) {
    ^bb0(%out: f32):
      %191 = linalg.index 0 : index
      %192 = linalg.index 1 : index
      %193 = arith.cmpi eq, %dim, %c1 : index
      %194 = arith.select %193, %c0, %191 : index
      %195 = arith.cmpi eq, %dim_16, %c1 : index
      %196 = arith.select %195, %c0, %192 : index
      %extracted = tensor.extract %179[%194, %196, %c0] : tensor<?x?x1xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x?x128xf32>
    cf.assert %150, "mismatched size for broadcast"
    cf.assert %151, "mismatched size for broadcast"
    %181 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%177, %180 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.subf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %182 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181, %181 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.mulf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %183 = linalg.generic {indexing_maps = [#map1, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%182 : tensor<?x?x128xf32>) outs(%141 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.addf %in, %out : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    %184 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%183 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.divf %in, %cst_15 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    %185 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%184 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = arith.truncf %cst_8 : f64 to f32
      %192 = arith.addf %in, %191 : f32
      linalg.yield %192 : f32
    } -> tensor<?x?x1xf32>
    %186 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%185 : tensor<?x?x1xf32>) outs(%140 : tensor<?x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %191 = math.rsqrt %in : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x1xf32>
    cf.assert %145, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %147, "unimplemented: dynamic negative broadcast sizes"
    %187 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%148 : tensor<?x?x128xf32>) {
    ^bb0(%out: f32):
      %191 = linalg.index 0 : index
      %192 = linalg.index 1 : index
      %193 = arith.cmpi eq, %dim, %c1 : index
      %194 = arith.select %193, %c0, %191 : index
      %195 = arith.cmpi eq, %dim_16, %c1 : index
      %196 = arith.select %195, %c0, %192 : index
      %extracted = tensor.extract %186[%194, %196, %c0] : tensor<?x?x1xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x?x128xf32>
    cf.assert %150, "mismatched size for broadcast"
    cf.assert %151, "mismatched size for broadcast"
    %188 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181, %187 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.mulf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %189 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %cst_3 : tensor<?x?x128xf32>, tensor<128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.mulf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    %190 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%189, %cst_2 : tensor<?x?x128xf32>, tensor<128xf32>) outs(%138 : tensor<?x?x128xf32>) {
    ^bb0(%in: f32, %in_56: f32, %out: f32):
      %191 = arith.addf %in, %in_56 : f32
      linalg.yield %191 : f32
    } -> tensor<?x?x128xf32>
    return %190 : tensor<?x?x128xf32>
  }
}

{-#

#-}
