// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @vector_chain_into_reduction(%arg0: tensor<4x8xf32>,
                                       %arg1: tensor<4x8xf32>,
                                       %arg2: tensor<4x8xf32>)
    -> tensor<4xf32> {
  %empty0 = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty0 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.addf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  %empty1 = tensor.empty() : tensor<4x8xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.mulf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  %empty2 = tensor.empty() : tensor<4xf32>
  %2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%1 : tensor<4x8xf32>)
    outs(%empty2 : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>

  return %2 : tensor<4xf32>
}

// CHECK: FusionCandidateAnalysis
// CHECK: primitive = "ElementwiseChain"
// CHECK-SAME: internal_ops = [0, 1]
// CHECK: primitive = "ReductionInlining"
// CHECK-SAME: internal_ops = [1, 2]
// CHECK: CandidateMergeAnalysis
// CHECK: merged_candidate_id = 0
// CHECK-SAME: source_candidates = [0, 1]
// CHECK-SAME: primary_ops = [0, 2]
// CHECK-SAME: internal_ops = [0, 1, 2]
// CHECK-SAME: primitive_combo = ["ElementwiseChain", "ReductionInlining"]
// CHECK-SAME: closed = true
// CHECK-SAME: families = ["reduction"]
