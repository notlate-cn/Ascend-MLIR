// RUN: not ascend-mlir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @unsupported_tensor_producer(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %generated = tensor.generate {
  ^bb0(%i: index):
    %v = arith.index_cast %i : index to i32
    %f = arith.sitofp %v : i32 to f32
    tensor.yield %f : f32
  } : tensor<4xf32>

  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%generated, %arg0 : tensor<4xf32>, tensor<4xf32>)
    outs(%arg0 : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>

  return %0 : tensor<4xf32>
}

// CHECK: error: unsupported Kernelize tensor producer "tensor.generate"
// CHECK-SAME: consumed by "linalg.generic"
