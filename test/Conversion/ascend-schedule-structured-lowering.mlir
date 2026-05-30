// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s
// RUN: rm -rf %t && mkdir -p %t && afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default debug-stage=schedule debug-dump-dir=%t' >/dev/null 2>/dev/null && FileCheck %s --input-file=%t/033-schedule-final.mlir --check-prefix=DUMP

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

// CHECK: DebugStepCatalog:
// CHECK:   step = "schedule.clear"
// CHECK:   title = "Clear stale schedule metadata"
// CHECK:   step = "schedule.choose-plan"
// CHECK:   outputs = "schedule candidates, selected decision, tile_params, tail_plan"
// CHECK:   step = "schedule.attach-contract"
// CHECK:   inspect_hint = "Primary ops and func attrs should expose the same decision_id, tile_params, tail_policies, and target_tile_policy."
// CHECK: ScheduleDecisionSet:
// CHECK-NEXT:   kernel = kernel_0
// CHECK: StructuredLowering:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   skeleton = loop_skeleton_v0
// CHECK-NEXT:   verified_ops = 2
// CHECK: ScheduleCache:
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// CHECK-SAME: ascend.schedule.guard_markers
// CHECK-SAME: kind = "shape_static_equal"
// CHECK-SAME: scope = "candidate"
// CHECK-SAME: text = "d0 == 64"
// CHECK-SAME: ascend.schedule.structured_lowering = "loop_skeleton_v0"
// CHECK-SAME: ascend.schedule.tail_markers
// CHECK-SAME: ascend.schedule.target_tile_policy = "target_default_32"

// DUMP: // Ascend DebugStep: schedule.attach-contract
// DUMP: // Purpose: Attach the chosen schedule decision as the downstream symbolic runtime contract.
// DUMP: // Outputs: decision_id, tile_params, tail_plan, tail_policies, target_tile_policy, structured lowering marker
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// CHECK-SAME: ascend.schedule.guard_markers
// CHECK-SAME: kind = "shape_static_equal"
// CHECK-SAME: scope = "candidate"
// CHECK-SAME: text = "d0 == 64"
// CHECK-SAME: ascend.schedule.structured_lowering = "loop_skeleton_v0"
// CHECK-SAME: ascend.schedule.tail_markers
// CHECK-SAME: ascend.schedule.target_tile_policy = "target_default_32"
