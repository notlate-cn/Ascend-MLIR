// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @elementwise_chain(%arg0: tensor<4x8xf16>,
                             %arg1: tensor<4x8xf16>,
                             %arg2: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty0 = tensor.empty() : tensor<4x8xf16>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty0 : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>

  %empty1 = tensor.empty() : tensor<4x8xf16>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %arg2 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty1 : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>

  return %1 : tensor<4x8xf16>
}

func.func @unsupported_unknown(%arg0: tensor<11xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0 + d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<11xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %o: f16):
    linalg.yield %x : f16
  } -> tensor<4x8xf16>

  return %0 : tensor<4x8xf16>
}

// CHECK: OpRoleClassification
// CHECK: op_id = 2
// CHECK-SAME: roles = ["Unsupported"]
// CHECK-SAME: op_role = "unsupported"
// CHECK: FusionCandidateAnalysis
// CHECK: candidate_id = 0
// CHECK-SAME: kind = "Fusion"
// CHECK-SAME: primitive = "ElementwiseChain"
// CHECK-SAME: primary_ops = [0]
// CHECK-SAME: internal_ops = [0, 1]
// CHECK-SAME: closed = true
// CHECK-SAME: benefit = 10
// CHECK-SAME: families = ["vector"]
// CHECK-NOT: primitive = "FallbackSingleOp"{{.*}}internal_ops = [2]
// CHECK: Kernelize report
