// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @softmax_like_reduction_into_vector(
    %arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty0 = tensor.empty() : tensor<4xf32>
  %max = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<4x8xf32>)
    outs(%empty0 : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %m = arith.maximumf %acc, %x : f32
    linalg.yield %m : f32
  } -> tensor<4xf32>

  %empty1 = tensor.empty() : tensor<4x8xf32>
  %sub = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %max : tensor<4x8xf32>, tensor<4xf32>)
    outs(%empty1 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %m: f32, %out: f32):
    %v = arith.subf %x, %m : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>

  return %sub : tensor<4x8xf32>
}

// CHECK: OpRoleClassification
// CHECK: op_id = 0
// CHECK-SAME: roles = ["Reduction"]
// CHECK-SAME: op_role = "reduction"
// CHECK: FusionCandidateAnalysis
// CHECK: primitive = "ConsumerIntoPrimary"
// CHECK-SAME: primary_ops = [1]
// CHECK-SAME: internal_ops = [0, 1]
// CHECK-SAME: families = ["vector"]
// CHECK: KernelPatternGraph
// CHECK: source = "Fusion"
// CHECK-SAME: internal_ops = [0, 1]
// CHECK-SAME: primary_ops = [1]
// CHECK: KernelPartition
// CHECK: kernel_pattern = "kernel_0"
// CHECK-SAME: internal_ops = [0, 1]
// CHECK-SAME: primary_ops = [1]
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK-SAME: ascend.op_role = "reduction"
// CHECK-NOT: ascend.primary = true
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK-SAME: ascend.op_role = "vector"
// CHECK-SAME: ascend.primary = true
