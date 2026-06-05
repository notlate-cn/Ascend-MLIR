// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @symbol_axis_contract_add_reduce(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %out: tensor<?xf32>)
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %n = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty0 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %add = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%empty0 : tensor<?x?xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>

  %red = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%add : tensor<?x?xf32>)
      outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %acc, %x : f32
      linalg.yield %v : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

// CHECK: ScheduleProblem:
// CHECK:   kernel = kernel_0
// CHECK:   tileable_axes = [arg0_dim0]
// CHECK:   required_reduction_axes = [arg0_dim1]
// CHECK:   axis_constraints = [
// CHECK:     axis=0 roles=[bind_core,kernel_loop,vectorize] tail=masked_tail sym=arg0_dim0
// CHECK:     axis=1 roles=[full_reduction] tail=full_extent sym=arg0_dim1
