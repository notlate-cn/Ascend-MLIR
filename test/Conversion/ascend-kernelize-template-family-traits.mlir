// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' 2>&1 | FileCheck %s

func.func @cube_vector_family(%arg0: tensor<4x4xf16>,
                              %arg1: tensor<4x4xf16>,
                              %rhs: tensor<4x4xf16>)
    -> tensor<4x4xf16> {
  %empty0 = tensor.empty() : tensor<4x4xf16>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x4xf16>, tensor<4x4xf16>)
    outs(%empty0 : tensor<4x4xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>

  %empty1 = tensor.empty() : tensor<4x4xf16>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %arg1 : tensor<4x4xf16>, tensor<4x4xf16>)
    outs(%empty1 : tensor<4x4xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>

  %empty2 = tensor.empty() : tensor<4x4xf16>
  %2 = linalg.matmul ins(%1, %rhs : tensor<4x4xf16>, tensor<4x4xf16>)
    outs(%empty2 : tensor<4x4xf16>) -> tensor<4x4xf16>

  %empty3 = tensor.empty() : tensor<4x4xf16>
  %3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%2 : tensor<4x4xf16>)
    outs(%empty3 : tensor<4x4xf16>) {
  ^bb0(%x: f16, %o: f16):
    linalg.yield %x : f16
  } -> tensor<4x4xf16>

  return %3 : tensor<4x4xf16>
}

// CHECK: CandidateMergeAnalysis
// CHECK: primitive_combo = ["ElementwiseChain", "ConsumerIntoPrimary"]
// CHECK-SAME: families = ["cube"]
// CHECK: family_resolver = "kernelize_trait_resolver"
