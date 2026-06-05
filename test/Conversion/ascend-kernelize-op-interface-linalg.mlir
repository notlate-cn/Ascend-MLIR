// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @generic_contraction(%arg0: tensor<4x8xf32>,
                               %arg1: tensor<8x16xf32>,
                               %arg2: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>)
    outs(%arg2 : tensor<4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op = "linalg.generic"
// CHECK-SAME: participation = "analyze"
// CHECK-SAME: model = "linalg_external"
// CHECK-SAME: traits = ["structured"]
// CHECK-SAME: access = "Contraction"
// CHECK-SAME: result_ranks = [2]
// CHECK-SAME: iterators = [parallel, parallel, reduction]
// CHECK: OpRoleClassification
// CHECK: op_id = 0
// CHECK-SAME: roles = ["Primary", "Vector"]
// CHECK-SAME: op_role = "vector"
