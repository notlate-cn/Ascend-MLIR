// RUN: afir-opt --afir-shape-inference %s | FileCheck %s

// Test shape inference pass

#map = affine_map<(d0, d1) -> (d0, d1)>


// CHECK-LABEL: func.func @test_shape_inference
func.func @test_shape_inference(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<?x?xf32> {
  // CHECK: afir.add {{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = VECTOR_IN, position_id = 0, depth = 0, is_double_buffer = false>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @test_chained_shape_inference
func.func @test_chained_shape_inference(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<?x?xf32> {
  // CHECK: afir.add {{.*}} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [8, 1], tensor_id = 0, reuse_id = -1, position = VECTOR_IN, position_id = 0, depth = 0, is_double_buffer = false>]} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK: afir.mul {{.*}} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = afir.mul %0, %arg1 {indexing_maps = [#map, #map, #map]}: (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
