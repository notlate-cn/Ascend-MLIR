// RUN: afir-opt %s | FileCheck %s

// Test basic AFIR operations

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @test_add
func.func @test_add(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: afir.add
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 8], tensor_id = 0, reuse_id = -1, position = vector_out, position_id = 0, depth = 0, is_double_buffer = false>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_sub
func.func @test_sub(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: afir.sub
  %0 = afir.sub %arg0, %arg1 {indexing_maps = [#map, #map, #map]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_mul
func.func @test_mul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: afir.mul
  %0 = afir.mul %arg0, %arg1 {indexing_maps = [#map, #map, #map]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_div
func.func @test_div(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: afir.div
  %0 = afir.div %arg0, %arg1 {indexing_maps = [#map, #map, #map]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_compound_ops
func.func @test_compound_ops(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>, %c: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: afir.mul
  // CHECK: afir.add
  %mul = afir.mul %a, %b {indexing_maps = [#map, #map, #map]} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %result = afir.add %mul, %c {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 8], tensor_id = 0, reuse_id = -1, position = vector_in, position_id = 0, depth = 0, is_double_buffer = false>]} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}
