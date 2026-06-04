// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @symbol_shape_constraints(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %out: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%out : tensor<?x?xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}

// CHECK: ScheduleProblem:
// CHECK:   shape_constraints = [d0 dynamic, d1 dynamic, dim_equal(arg0_dim0), dim_equal(arg0_dim1)]
