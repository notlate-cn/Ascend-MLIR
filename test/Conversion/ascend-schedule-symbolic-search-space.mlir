// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @symbolic_dynamic_vector(%arg0: tensor<?x?xf16>,
                                   %arg1: tensor<?x?xf16>,
                                   %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
  return %add : tensor<?x?xf16>
}

// CHECK: ScheduleSearch:
// CHECK:   generated =
// CHECK:   kept = 4
// CHECK:   compile_time_top_k = 4
// CHECK: ScheduleGuards:
// CHECK:   candidate_guard = T_arg0_dim0 > 0
// CHECK:   candidate_guard = T_arg0_dim0 <= arg0_dim0
// CHECK:   candidate_guard = T_arg0_dim1 > 0
// CHECK:   candidate_guard = T_arg0_dim1 <= arg0_dim1
// CHECK: ScheduleDecisionSet:
// CHECK:   tile_params = [name=T_arg0_dim0 axis=0 binding=runtime
// CHECK-SAME: [name=T_arg0_dim1 axis=1 binding=runtime
