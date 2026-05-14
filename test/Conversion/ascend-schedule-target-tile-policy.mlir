// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=target-aware cann-root=%S/Inputs/ascend-schedule-target-tile-cann soc=SyntheticScheduleSoC dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @dynamic_vector(%arg0: tensor<?x128xf16>,
                          %arg1: tensor<?x128xf16>) -> tensor<?x128xf16> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x128xf16>
  %empty = tensor.empty(%d0) : tensor<?x128xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x128xf16>, tensor<?x128xf16>)
    outs(%empty : tensor<?x128xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<?x128xf16>
  return %out : tensor<?x128xf16>
}

// CHECK: ScheduleDecisionSet:
// CHECK:   selected_tile_shape = [64,128]
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.selected_tile_shape = array<i64: 64, 128>
// CHECK-SAME: ascend.schedule.target_tile_policy = "target_ub_64"
