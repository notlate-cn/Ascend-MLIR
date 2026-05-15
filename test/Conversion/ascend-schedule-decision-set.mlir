// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule runtime-top-k=2' 2>&1 | FileCheck %s --check-prefix=RUNTIME-TOP-K-2
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule runtime-top-k=0' 2>&1 | FileCheck %s --check-prefix=RUNTIME-TOP-K-ZERO

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

// DEFAULT: ScheduleDecisionSet:
// DEFAULT-NEXT:   kernel = kernel_0
// DEFAULT-NEXT:   decisions = 2
// DEFAULT-NEXT:   runtime_top_k = 1
// DEFAULT-NEXT:   selected = kernel_0.decision.0
// DEFAULT: schedule_decision_id = "kernel_0.decision.0"
// DEFAULT: linalg.generic
// DEFAULT-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// DEFAULT-SAME: ascend.schedule.runtime_top_k = 1 : i64
// DEFAULT: linalg.generic
// DEFAULT-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// DEFAULT-SAME: ascend.schedule.runtime_top_k = 1 : i64

// RUNTIME-TOP-K-2: ScheduleDecisionSet:
// RUNTIME-TOP-K-2-NEXT:   kernel = kernel_0
// RUNTIME-TOP-K-2-NEXT:   decisions = 2
// RUNTIME-TOP-K-2-NEXT:   runtime_top_k = 2
// RUNTIME-TOP-K-2-NEXT:   selected = kernel_0.decision.0
// RUNTIME-TOP-K-2: linalg.generic
// RUNTIME-TOP-K-2-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// RUNTIME-TOP-K-2-SAME: ascend.schedule.runtime_top_k = 2 : i64
// RUNTIME-TOP-K-2: linalg.generic
// RUNTIME-TOP-K-2-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// RUNTIME-TOP-K-2-SAME: ascend.schedule.runtime_top_k = 2 : i64

// RUNTIME-TOP-K-ZERO: ScheduleDecisionSet:
// RUNTIME-TOP-K-ZERO-NEXT:   kernel = kernel_0
// RUNTIME-TOP-K-ZERO-NEXT:   decisions = 2
// RUNTIME-TOP-K-ZERO-NEXT:   runtime_top_k = 1
// RUNTIME-TOP-K-ZERO-NEXT:   selected = kernel_0.decision.0
// RUNTIME-TOP-K-ZERO: linalg.generic
// RUNTIME-TOP-K-ZERO-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// RUNTIME-TOP-K-ZERO-SAME: ascend.schedule.runtime_top_k = 1 : i64
// RUNTIME-TOP-K-ZERO: linalg.generic
// RUNTIME-TOP-K-ZERO-SAME: ascend.schedule.decision_id = "kernel_0.decision.0"
// RUNTIME-TOP-K-ZERO-SAME: ascend.schedule.runtime_top_k = 1 : i64
