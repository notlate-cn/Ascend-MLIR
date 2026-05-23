// RUN: sed -n '/\/\/ MERGE-BEGIN/,/\/\/ MERGE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=MERGE
// RUN: sed -n '/\/\/ REVERSE-BEGIN/,/\/\/ REVERSE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=REVERSE
// RUN: sed -n '/\/\/ SUBSUMED-BEGIN/,/\/\/ SUBSUMED-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=SUBSUMED
// RUN: sed -n '/\/\/ HORIZONTAL-BEGIN/,/\/\/ HORIZONTAL-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=HORIZONTAL
// RUN: sed -n '/\/\/ HORIZONTAL-SHAPE-BEGIN/,/\/\/ HORIZONTAL-SHAPE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: sed -n '/\/\/ OUTS-BEGIN/,/\/\/ OUTS-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=OUTS
// RUN: sed -n '/\/\/ DEPENDENCY-BEGIN/,/\/\/ DEPENDENCY-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s --check-prefix=DEPENDENCY

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
func.func @vector_chain_into_cube(%arg0: tensor<4x4xf16>,
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
// REVERSE-END

// SUBSUMED-BEGIN
func.func @nested_vector_chain(%arg0: tensor<4x8xf32>,
                               %arg1: tensor<4x8xf32>,
                               %arg2: tensor<4x8xf32>,
                               %arg3: tensor<4x8xf32>) -> tensor<4x8xf32> {
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

  %empty2 = tensor.empty() : tensor<4x8xf32>
  %2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%1, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty2 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.addf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  return %2 : tensor<4x8xf32>
}
// SUBSUMED-END

// HORIZONTAL-BEGIN
func.func @horizontal_siblings(%arg0: tensor<4x8xf32>,
                               %arg1: tensor<4x8xf32>,
                               %arg2: tensor<4x8xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
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
  } ins(%arg0, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.mulf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}
// HORIZONTAL-END

// HORIZONTAL-SHAPE-BEGIN
func.func @horizontal_fill_shape_mismatch(%arg0: tensor<4x8xf32>,
                                          %arg1: tensor<4x16xf32>)
    -> (tensor<4x8xf32>, tensor<4x16xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %empty0 = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.fill ins(%cst : f32)
                   outs(%empty0 : tensor<4x8xf32>) -> tensor<4x8xf32>

  %empty1 = tensor.empty() : tensor<4x16xf32>
  %1 = linalg.fill ins(%cst : f32)
                   outs(%empty1 : tensor<4x16xf32>) -> tensor<4x16xf32>

  return %0, %1 : tensor<4x8xf32>, tensor<4x16xf32>
}
// HORIZONTAL-SHAPE-END

// OUTS-BEGIN
func.func @shared_outs_init_not_horizontal(%arg0: tensor<4x8xf32>,
                                           %arg1: tensor<4x8xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %empty = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>

  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg1 : tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>

  return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}
// OUTS-END

// DEPENDENCY-BEGIN
func.func @passthrough_dependency_not_horizontal(%arg0: tensor<4x8xf32>,
                                                 %arg1: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
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

  %slice = tensor.extract_slice %0[0, 0] [4, 8] [1, 1]
      : tensor<4x8xf32> to tensor<4x8xf32>

  %empty1 = tensor.empty() : tensor<4x8xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%slice, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %v = arith.mulf %x, %y : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  return %1 : tensor<4x8xf32>
}
// DEPENDENCY-END

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

// REVERSE: FusionCandidateAnalysis
// REVERSE: primitive = "ElementwiseChain"
// REVERSE-SAME: internal_ops = [0, 1]
// REVERSE: primitive = "ConsumerIntoPrimary"
// REVERSE-SAME: internal_ops = [2, 3]
// REVERSE: CandidateMergeAnalysis
// REVERSE: merged_candidate_id = 0
// REVERSE-SAME: source_candidates = [0, 1]
// REVERSE-SAME: primary_ops = [0, 2]
// REVERSE-SAME: internal_ops = [0, 1, 2, 3]
// REVERSE-SAME: primitive_combo = ["ElementwiseChain", "ConsumerIntoPrimary"]
// REVERSE-SAME: closed = true
// REVERSE-SAME: families = ["cube"]
// REVERSE-NOT: merged_candidate_id =
// REVERSE: Kernelize report

// SUBSUMED: FusionCandidateAnalysis
// SUBSUMED: primitive = "ElementwiseChain"
// SUBSUMED-SAME: internal_ops = [0, 1, 2]
// SUBSUMED: primitive = "ElementwiseChain"
// SUBSUMED-SAME: internal_ops = [1, 2]
// SUBSUMED: CandidateMergeAnalysis
// SUBSUMED-NOT: merged_candidate_id =
// SUBSUMED: Kernelize report

// HORIZONTAL: FusionCandidateAnalysis
// HORIZONTAL: primitive = "FallbackSingleOp"
// HORIZONTAL-SAME: internal_ops = [0]
// HORIZONTAL: primitive = "FallbackSingleOp"
// HORIZONTAL-SAME: internal_ops = [1]
// HORIZONTAL: CandidateMergeAnalysis
// HORIZONTAL-NOT: merged_candidate_id =
// HORIZONTAL: HorizontalFusionAnalysis
// HORIZONTAL: horizontal_candidate_id = 0
// HORIZONTAL-SAME: sibling_candidates = [0, 1]
// HORIZONTAL-SAME: shared_inputs = 1
// HORIZONTAL-SAME: per_group_contracts = 2
// HORIZONTAL-SAME: benefit = 15
// HORIZONTAL: Kernelize report

// SHAPE: HorizontalFusionAnalysis
// SHAPE-NOT: horizontal_candidate_id =
// SHAPE: Kernelize report

// OUTS: FusionCandidateAnalysis
// OUTS: primitive = "FallbackSingleOp"
// OUTS-SAME: internal_ops = [0]
// OUTS: primitive = "FallbackSingleOp"
// OUTS-SAME: internal_ops = [1]
// OUTS: HorizontalFusionAnalysis
// OUTS-NOT: horizontal_candidate_id =
// OUTS: Kernelize report

// DEPENDENCY: FusionCandidateAnalysis
// DEPENDENCY: primitive = "FallbackSingleOp"
// DEPENDENCY-SAME: internal_ops = [0]
// DEPENDENCY: primitive = "FallbackSingleOp"
// DEPENDENCY-SAME: internal_ops = [1]
// DEPENDENCY: HorizontalFusionAnalysis
// DEPENDENCY-NOT: horizontal_candidate_id =
// DEPENDENCY: Kernelize report
