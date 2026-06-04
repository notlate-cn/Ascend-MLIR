// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @dynamic_reduction(%arg0: tensor<?x?xf16>) -> tensor<?xf16> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %empty = tensor.empty(%m) : tensor<?xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<?x?xf16>)
    outs(%empty : tensor<?xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<?xf16>
  return %out : tensor<?xf16>
}

// CHECK: ScheduleDecisionSet:
// CHECK:   kernel = kernel_0
// CHECK:   selected = kernel_0.decision.0
// CHECK-NEXT:   candidate_guards = 2
// CHECK-NEXT:   decision_guards = 0
// CHECK-NEXT:   tile_params = [name=T_arg0_dim0 axis=0 binding=runtime
