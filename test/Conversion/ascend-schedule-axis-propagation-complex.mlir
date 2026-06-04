// RUN: afir-opt %s --split-input-file --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @transpose_preserves_symbol_axes(%arg0: tensor<?x?xf16>,
                                           %out: tensor<?x?xf16>)
                                           -> tensor<?x?xf16> {
  %t = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>) {
    ^bb0(%x: f16, %o: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  return %t : tensor<?x?xf16>
}

// CHECK: ScheduleProblem:
// CHECK: tileable_axes = [arg0_dim1, arg0_dim0]

// -----

func.func @split_axis_barrier(%arg0: tensor<?x8xf16>,
                              %out: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x8xf16>
  %slice = tensor.extract_slice %arg0[0, 0] [%m, 8] [1, 1] :
      tensor<?x8xf16> to tensor<?x8xf16>
  %copy = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%slice : tensor<?x8xf16>)
      outs(%out : tensor<?x8xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.primary = true
      } {
    ^bb0(%x: f16, %o: f16):
      linalg.yield %x : f16
    } -> tensor<?x8xf16>
  return %copy : tensor<?x8xf16>
}

// CHECK: ScheduleProblem:
// CHECK: structure_constraints = [split_axis_barrier

// -----

func.func @concat_dim0_barrier(%a: tensor<?x8xf16>, %b: tensor<?x8xf16>,
                               %out: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %c = tensor.concat dim(0) %a, %b :
      (tensor<?x8xf16>, tensor<?x8xf16>) -> tensor<?x8xf16>
  %copy = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%c : tensor<?x8xf16>)
      outs(%out : tensor<?x8xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.primary = true
      } {
    ^bb0(%x: f16, %o: f16):
      linalg.yield %x : f16
    } -> tensor<?x8xf16>
  return %copy : tensor<?x8xf16>
}

// CHECK: ScheduleProblem:
// CHECK: structure_constraints = [concat_axis_barrier

// -----

func.func @reshape_static_bridge(%arg0: tensor<?x8xf16>,
                                 %out: tensor<?xf16>) -> tensor<?xf16> {
  %flat = tensor.collapse_shape %arg0 [[0, 1]] :
      tensor<?x8xf16> into tensor<?xf16>
  %copy = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%flat : tensor<?xf16>)
      outs(%out : tensor<?xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.primary = true
      } {
    ^bb0(%x: f16, %o: f16):
      linalg.yield %x : f16
    } -> tensor<?xf16>
  return %copy : tensor<?xf16>
}

// CHECK: ScheduleProblem:
// CHECK: structure_constraints = [reshape_static_bridge

// -----

func.func @branch_merge_axes_consistent(%arg0: tensor<?x?xf16>,
                                        %arg1: tensor<?x?xf16>,
                                        %out: tensor<?x?xf16>)
                                        -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %n = tensor.dim %arg0, %c1 : tensor<?x?xf16>
  %empty0 = tensor.empty(%m, %n) : tensor<?x?xf16>
  %left = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf16>)
      outs(%empty0 : tensor<?x?xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.branch_group = 0 : i64
      } {
    ^bb0(%x: f16, %o: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  %empty1 = tensor.empty(%m, %n) : tensor<?x?xf16>
  %right = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<?x?xf16>)
      outs(%empty1 : tensor<?x?xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.branch_group = 1 : i64
      } {
    ^bb0(%x: f16, %o: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  %merge = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%left, %right : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "vector",
        ascend.merge_group = 0 : i64,
        ascend.primary = true
      } {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>
  return %merge : tensor<?x?xf16>
}

// CHECK: ScheduleProblem:
// CHECK: structure_constraints = [branch_merge_axes_consistent

// -----

func.func @softmax_two_reductions_share_tile_axis(%arg0: tensor<?x?xf32>,
                                                  %out: tensor<?x?xf32>)
                                                  -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %n = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty0 = tensor.empty(%m) : tensor<?xf32>
  %max = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%empty0 : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.maximumf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<?xf32>
  %empty1 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %sub = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %max : tensor<?x?xf32>, tensor<?xf32>)
      outs(%empty1 : tensor<?x?xf32>) {
    ^bb0(%x: f32, %mval: f32, %o: f32):
      %v = arith.subf %x, %mval : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
  %sum = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%sub : tensor<?x?xf32>)
      outs(%empty0 : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %v = arith.addf %x, %acc : f32
      linalg.yield %v : f32
    } -> tensor<?xf32>
  %div = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%sub, %sum : tensor<?x?xf32>, tensor<?xf32>)
      outs(%out : tensor<?x?xf32>) {
    ^bb0(%x: f32, %s: f32, %o: f32):
      %v = arith.divf %x, %s : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
  return %div : tensor<?x?xf32>
}

// CHECK: ScheduleProblem:
// CHECK: structure_constraints = [single_reduction_region, multi_reduction_consistent
// CHECK: tileable_axes = [arg0_dim0]
// CHECK: required_reduction_axes = [arg0_dim1]
