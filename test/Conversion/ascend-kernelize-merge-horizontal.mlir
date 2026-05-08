// RUN: sed -n '/\/\/ MERGE-BEGIN/,/\/\/ MERGE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=MERGE
// RUN: sed -n '/\/\/ REVERSE-BEGIN/,/\/\/ REVERSE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=REVERSE

// MERGE-BEGIN
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
// MERGE-END

// REVERSE-BEGIN
func.func @vector_chain_into_cube(%arg0: tensor<4x4xf32>,
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
// REVERSE-END

// MERGE: FusionCandidateAnalysis
// MERGE: primitive = "ElementwiseChain"
// MERGE-SAME: internal_ops = [0, 1]
// MERGE: primitive = "ReductionInlining"
// MERGE-SAME: internal_ops = [1, 2]
// MERGE: CandidateMergeAnalysis
// MERGE: merged_candidate_id = 0
// MERGE-SAME: source_candidates = [0, 1]
// MERGE-SAME: primary_ops = [0, 2]
// MERGE-SAME: internal_ops = [0, 1, 2]
// MERGE-SAME: primitive_combo = ["ElementwiseChain", "ReductionInlining"]
// MERGE-SAME: closed = true
// MERGE-SAME: families = ["reduction"]
// MERGE-NOT: merged_candidate_id =
// MERGE: Kernelize report

// REVERSE: CandidateMergeAnalysis
// REVERSE-NOT: merged_candidate_id =
// REVERSE: Kernelize report
