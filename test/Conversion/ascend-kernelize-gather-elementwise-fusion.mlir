// RUN: sed -n '/\/\/ VALID-BEGIN/,/\/\/ VALID-END/p' %s | ascend-mlir-opt --ascend-normalize --ascend-kernelize | FileCheck %s --check-prefix=VALID
// RUN: sed -n '/\/\/ REDUCTION-BEGIN/,/\/\/ REDUCTION-END/p' %s | ascend-mlir-opt --ascend-normalize --ascend-kernelize | FileCheck %s --check-prefix=REDUCTION
// RUN: sed -n '/\/\/ PREMAP-BEGIN/,/\/\/ PREMAP-END/p' %s | ascend-mlir-opt --ascend-normalize --ascend-kernelize | FileCheck %s --check-prefix=PREMAP

// VALID-BEGIN
#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>

module {
  func.func @valid_relu_gather_add(%data: tensor<?x?xf16>,
                                   %indices: tensor<?xi64>,
                                   %bias: tensor<?xf16>)
      -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16
    %dim_m = tensor.dim %data, %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data, %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>
    %empty_relu = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %relu_out = linalg.generic {
      indexing_maps = [#id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%data : tensor<?x?xf16>) outs(%empty_relu : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi64>) outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %out: f16):
      %i = linalg.index 0 : index
      %idx_cast = arith.index_cast %idx : i64 to index
      %val = tensor.extract %relu_out[%i, %idx_cast] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?xf16>
    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#id, #col, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%gathered, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%g: f16, %b: f16, %o: f16):
      %v = arith.addf %g, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
    return %out : tensor<?x?xf16>
  }
}
// VALID-END

// VALID-LABEL: func.func @valid_relu_gather_add
// VALID: linalg.generic
// VALID-SAME: ins(%{{.*}}, %{{.*}} : tensor<?xi64>, tensor<?xf16>)
// VALID-SAME: gather_dim = 1
// VALID: tensor.extract %{{.*}}[
// VALID-SAME: : tensor<?x?xf16>
// VALID: arith.maximumf
// VALID: arith.addf
// VALID-NOT: linalg.generic
// VALID: return

// REDUCTION-BEGIN
#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>

module {
  func.func @gather_reduction_post_not_fused(%data: tensor<?x?xf16>,
                                             %indices: tensor<?xi64>)
      -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16
    %dim_m = tensor.dim %data, %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data, %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>
    %empty_relu = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %relu_out = linalg.generic {
      indexing_maps = [#id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%data : tensor<?x?xf16>) outs(%empty_relu : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi64>) outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %out: f16):
      %i = linalg.index 0 : index
      %idx_cast = arith.index_cast %idx : i64 to index
      %val = tensor.extract %relu_out[%i, %idx_cast] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?xf16>
    %empty_out = tensor.empty(%dim_m) : tensor<?xf16>
    %out = linalg.generic {
      indexing_maps = [#id, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%gathered : tensor<?x?xf16>) outs(%empty_out : tensor<?xf16>) {
    ^bb0(%g: f16, %acc: f16):
      %v = arith.addf %g, %acc : f16
      linalg.yield %v : f16
    } -> tensor<?xf16>
    return %out : tensor<?xf16>
  }
}
// REDUCTION-END

// REDUCTION-LABEL: func.func @gather_reduction_post_not_fused
// REDUCTION: linalg.generic
// REDUCTION-SAME: gather_dim = 1
// REDUCTION: arith.maximumf
// REDUCTION: linalg.generic
// REDUCTION-SAME: iterator_types = ["parallel", "reduction"]
// REDUCTION: return

// PREMAP-BEGIN
#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

module {
  func.func @transposed_pre_not_fused(%data: tensor<?x?xf16>,
                                      %indices: tensor<?xi64>)
      -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim_m = tensor.dim %data, %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data, %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>
    %empty_transposed = tensor.empty(%dim_n, %dim_m) : tensor<?x?xf16>
    %transposed = linalg.generic {
      indexing_maps = [#transpose, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%data : tensor<?x?xf16>) outs(%empty_transposed : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?xf16>
    %empty_gathered = tensor.empty(%dim_n, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi64>) outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %out: f16):
      %i = linalg.index 0 : index
      %idx_cast = arith.index_cast %idx : i64 to index
      %val = tensor.extract %transposed[%i, %idx_cast] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?xf16>
    return %gathered : tensor<?x?xf16>
  }
}
// PREMAP-END

// PREMAP-LABEL: func.func @transposed_pre_not_fused
// PREMAP: %[[TRANSPOSED:.*]] = linalg.generic
// PREMAP: linalg.generic
// PREMAP-SAME: gather_dim = 1
// PREMAP: tensor.extract %[[TRANSPOSED]]
// PREMAP: return
