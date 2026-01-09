// RUN: afir-opt --afir-shape-inference %s | FileCheck %s

// Test ShapeHelperOpInterface

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @test_binary_shape_helper
func.func @test_binary_shape_helper(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // Shape helper should infer output shape from inputs
  // CHECK: afir.add {{.*}} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1, 2], vectorized_strides = [12, 4, 1], tensor_id = 0, reuse_id = -1, position = vector_in, position_id = 0, depth = 0, is_double_buffer = false>]} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @test_shape_helper_chain
func.func @test_shape_helper_chain(%a: tensor<5x5xf32>, %b: tensor<5x5xf32>, %c: tensor<5x5xf32>) -> tensor<5x5xf32> {
  // CHECK: afir.sub {{.*}} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %0 = afir.sub %a, %b {indexing_maps = [#map, #map, #map]} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: afir.mul {{.*}} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %1 = afir.mul %0, %0 {indexing_maps = [#map, #map, #map]} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: afir.div {{.*}} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %2 = afir.div %1, %c {indexing_maps = [#map, #map, #map]} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  return %2 : tensor<5x5xf32>
}
