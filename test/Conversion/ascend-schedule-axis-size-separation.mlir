// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @same_size_distinct_axes(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %empty = tensor.empty(%n, %n) : tensor<?x?xf16>
  %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf16>)
      outs(%empty : tensor<?x?xf16>) {
    ^bb0(%x: f16, %o: f16):
      %v = arith.addf %x, %o : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
  return %out : tensor<?x?xf16>
}

// CHECK: ScheduleProblem:
// CHECK:   kernel = kernel_0
// CHECK:   shape_constraints = [d0 dynamic, d1 dynamic, dim_equal(arg0_dim0)]
// CHECK:   tileable_axes = [axis0(sym=arg0_dim0), axis1(sym=arg0_dim0)]
// CHECK:   axis_constraints = [
// CHECK:     axis=0 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail sym=arg0_dim0
// CHECK:     axis=1 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail sym=arg0_dim0
// CHECK: ScheduleGuards:
// CHECK:   candidate_guard = T_axis0_arg0_dim0 > 0
// CHECK:   candidate_guard = T_axis0_arg0_dim0 <= arg0_dim0
// CHECK:   candidate_guard = T_axis1_arg0_dim0 > 0
// CHECK:   candidate_guard = T_axis1_arg0_dim0 <= arg0_dim0
// CHECK: ScheduleDecisionSet:
// CHECK:   tile_params = [name=T_axis0_arg0_dim0 axis=0 binding=runtime
// CHECK-SAME: [name=T_axis1_arg0_dim0 axis=1 binding=runtime
