// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' 2>&1 | FileCheck %s

func.func @cube_vector_family(%arg0: tensor<4x4xf32>,
                              %arg1: tensor<4x4xf32>,
                              %rhs: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %empty0 = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%empty0 : tensor<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.addf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x4xf32>

  %empty1 = tensor.empty() : tensor<4x4xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%empty1 : tensor<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.mulf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x4xf32>

  %empty2 = tensor.empty() : tensor<4x4xf32>
  %2 = linalg.matmul ins(%1, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%empty2 : tensor<4x4xf32>) -> tensor<4x4xf32>

  %empty3 = tensor.empty() : tensor<4x4xf32>
  %3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%2 : tensor<4x4xf32>)
    outs(%empty3 : tensor<4x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<4x4xf32>

  return %3 : tensor<4x4xf32>
}

// CHECK: CandidateMergeAnalysis
// CHECK: primitive_combo = ["ElementwiseChain", "ConsumerIntoPrimary"]
// CHECK-SAME: families = ["cube"]
// CHECK: family_resolver = "kernelize_trait_resolver"
