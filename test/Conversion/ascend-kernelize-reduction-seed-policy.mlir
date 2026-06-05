// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' 2>&1 | FileCheck %s

func.func @row_sum_then_vector(%arg0: tensor<32x64xf32>,
                               %arg1: tensor<32xf32>,
                               %out: tensor<32xf32>) -> tensor<32xf32> {
  %sum = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<32x64xf32>)
    outs(%arg1 : tensor<32xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %r = arith.addf %acc, %a : f32
    linalg.yield %r : f32
  } -> tensor<32xf32>

  %scaled = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%sum : tensor<32xf32>)
    outs(%out : tensor<32xf32>) {
  ^bb0(%a: f32, %o: f32):
    %c = arith.constant 2.000000e+00 : f32
    %r = arith.mulf %a, %c : f32
    linalg.yield %r : f32
  } -> tensor<32xf32>
  return %scaled : tensor<32xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: access = "Reduction"
// CHECK-SAME: seed_policy = "non_seed_when_fused"
// CHECK: OpRoleClassification
// CHECK: op_id = 0 roles = ["Reduction"]
// CHECK: op_id = 1 roles = ["Primary", "Vector", "Injective"]
// CHECK: FusionCandidateAnalysis
// CHECK: primitive = "ConsumerIntoPrimary"
// CHECK-SAME: primary_ops = [1]
// CHECK-SAME: internal_ops = [0, 1]
// CHECK: KernelPartition
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
