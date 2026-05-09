// RUN: not afir-opt %s --ascend-realize 2>&1 | FileCheck %s

func.func @partial_schedule_attrs(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty0 = tensor.empty() : tensor<64xf16>
  %valid = linalg.generic {
    ascend.kernel = "kernel_0",
    ascend.schedule.decision_id = "kernel_0.decision.0",
    ascend.schedule.structured_lowering = "loop_skeleton_v0",
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
  %partial = linalg.generic {
    ascend.schedule.decision_id = "kernel_1.decision.0",
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%valid, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty1 : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %partial : tensor<64xf16>
}

// CHECK: error: ascend-realize requires complete scheduled structured lowering attributes
