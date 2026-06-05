// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' 2>&1 | FileCheck %s

func.func @native_interface_compat(%arg0: tensor<8xf32>,
                                   %arg1: tensor<8xf32>,
                                   %out: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%out : tensor<8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %r = arith.addf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op = "linalg.generic"
// CHECK-SAME: model = "linalg_external"
// CHECK: KernelPartition
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
