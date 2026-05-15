// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @view_chain(%arg0: tensor<4x16xf32>,
                      %arg1: tensor<4x16xf32>,
                      %arg2: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4x16xf32>)
    outs(%arg0 : tensor<4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4x16xf32>

  %1 = tensor.collapse_shape %0 [[0, 1]]
      : tensor<4x16xf32> into tensor<64xf32>
  %2 = tensor.cast %1 : tensor<64xf32> to tensor<64xf32>

  %3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%2, %arg2 : tensor<64xf32>, tensor<64xf32>)
    outs(%arg2 : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<64xf32>

  return %3 : tensor<64xf32>
}

func.func @independent_linalg_ops(
    %arg0: tensor<16xf32>, %arg1: tensor<16xf32>,
    %arg2: tensor<16xf32>) -> tensor<16xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>)
    outs(%arg2 : tensor<16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<16xf32>

  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg2 : tensor<16xf32>, tensor<16xf32>)
    outs(%arg2 : tensor<16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<16xf32>

  return %1 : tensor<16xf32>
}

func.func @reshape_view_chain(%arg0: tensor<4x8xf32>,
                              %arg1: tensor<8x4xf32>,
                              %arg2: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<4x8xf32>)
    outs(%arg0 : tensor<4x8xf32>) {
  ^bb0(%a: f32, %out: f32):
    linalg.yield %a : f32
  } -> tensor<4x8xf32>

  %c8_i64 = arith.constant 8 : i64
  %c4_i64 = arith.constant 4 : i64
  %shape = tensor.from_elements %c8_i64, %c4_i64 : tensor<2xi64>
  %reshaped = tensor.reshape %0(%shape)
      : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<8x4xf32>

  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%reshaped, %arg1 : tensor<8x4xf32>, tensor<8x4xf32>)
    outs(%arg2 : tensor<8x4xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<8x4xf32>

  return %1 : tensor<8x4xf32>
}

func.func @constant_transpose_source() -> tensor<8x4xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.transpose ins(%cst : tensor<4x8xf32>)
      outs(%empty : tensor<8x4xf32>) permutation = [1, 0]
  return %0 : tensor<8x4xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: producers = 0
// CHECK-SAME: consumers = 1
// CHECK: op_id = 1
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: producers = 1
// CHECK-SAME: consumers = 0
// CHECK: op_id = 2
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: producers = 0
// CHECK-SAME: consumers = 0
// CHECK: op_id = 3
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: producers = 0
// CHECK-SAME: consumers = 0
// CHECK: op_id = 4
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: producers = 0
// CHECK-SAME: consumers = 1
// CHECK: op_id = 5
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: producers = 1
// CHECK-SAME: consumers = 0
// CHECK: op_id = 6
// CHECK-SAME: op = "linalg.transpose"
// CHECK-SAME: producers = 0
// CHECK: KernelPartition
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK: tensor.collapse_shape
// CHECK-NOT: ascend.kernel
// CHECK: tensor.cast
// CHECK-NOT: ascend.kernel
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
