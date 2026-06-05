// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @fill_tensor(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32)
      outs(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: op = "linalg.fill"
// CHECK-SAME: participation = "analyze"
// CHECK-SAME: model = "linalg_external"
// CHECK-SAME: traits = ["structured"]
// CHECK-SAME: access = "Elementwise"
// CHECK-SAME: result_ranks = [2]
// CHECK-SAME: iterators = [parallel, parallel]
// CHECK: OpRoleClassification
// CHECK: op_id = 0
// CHECK-SAME: roles = ["Primary", "Vector", "Injective"]
// CHECK-SAME: op_role = "vector"
// CHECK: KernelPartition
// CHECK: kernel_pattern = "kernel_0"
// CHECK-SAME: internal_ops = [0]
// CHECK-SAME: primary_ops = [0]
// CHECK: linalg.fill
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK-SAME: ascend.op_role = "vector"
