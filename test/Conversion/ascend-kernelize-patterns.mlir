// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @pattern_partition(%arg0: tensor<16xf32>,
                             %arg1: tensor<16xf32>,
                             %arg2: tensor<16xf32>) -> tensor<16xf32> {
  %empty0 = tensor.empty() : tensor<16xf32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>)
    outs(%empty0 : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  } -> tensor<16xf32>

  %empty1 = tensor.empty() : tensor<16xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<16xf32>)
    outs(%empty1 : tensor<16xf32>) {
  ^bb0(%x: f32, %out: f32):
    %scale = arith.mulf %x, %x : f32
    linalg.yield %scale : f32
  } -> tensor<16xf32>

  return %1 : tensor<16xf32>
}

// CHECK: KernelPatternGraph
// CHECK: pattern_candidate_id = 0
// CHECK-SAME: source = "Fusion"
// CHECK-SAME: internal_ops = [0, 1]
// CHECK-SAME: primary_ops = [0]
// CHECK: KernelPartition
// CHECK: kernel_pattern = "kernel_0"
// CHECK-SAME: internal_ops = [0, 1]
// CHECK-SAME: primary_ops = [0]
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK-SAME: ascend.primary = true
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
