// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @elementwise_chain(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>,
                             %arg2: tensor<64xf16>) -> tensor<64xf16> {
  %empty0 = tensor.empty() : tensor<64xf16>
  %add = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty0 : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  %empty1 = tensor.empty() : tensor<64xf16>
  %mul = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%add, %arg2 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty1 : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  return %mul : tensor<64xf16>
}

// CHECK: SchedulePatternView:
// CHECK: kernel = kernel_0
// CHECK: ops = 2
// CHECK: primary_ops = 1
// CHECK: dominant_role = vector
// CHECK: linalg.generic
// CHECK-SAME: ascend.v2.schedule.decision_id = "decision_0"
// CHECK: linalg.generic
// CHECK-SAME: ascend.v2.schedule.decision_id = "decision_0"
// CHECK: ascend.v2.schedule.family = "vector_static_1d"
