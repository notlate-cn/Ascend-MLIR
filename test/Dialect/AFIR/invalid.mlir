// RUN: afir-opt %s -verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1d = affine_map<(d0) -> (d0)>

// ========================================================================
// Verification Error Tests - Type Mismatches
// ========================================================================

func.func @test_unary_rank_mismatch(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{'afir.abs' op requires the same type for all operands and results}}
  %0 = afir.abs %arg0 {indexing_maps = [#map, #map1d], outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @test_unary_shape_mismatch(%arg0: tensor<4x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{'afir.abs' op requires the same type for all operands and results}}
  %0 = afir.abs %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [2, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @test_binary_rank_mismatch(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{'afir.add' op operands must have the same rank}}
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map1d, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @test_binary_shape_mismatch(%arg0: tensor<4x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{'afir.add' op operands must have compatible shapes}}
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Verification Error Tests - Concat Operation
// ========================================================================

func.func @test_concat_axis_out_of_bounds(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<5x4xf32> {
  // expected-error @+1 {{'afir.concat' op axis 2 is out of bounds}}
  %0 = afir.concat %arg0, %arg1 {indexing_maps = [#map, #map, #map], concat_axis = 2 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4xf32>, tensor<3x4xf32> -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

func.func @test_concat_negative_axis(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<5x4xf32> {
  // expected-error @+1 {{'afir.concat' op axis -1 is out of bounds}}
  %0 = afir.concat %arg0, %arg1 {indexing_maps = [#map, #map, #map], concat_axis = -1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4xf32>, tensor<3x4xf32> -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

func.func @test_concat_rank_mismatch(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4x4xf32>) -> tensor<5x4x4xf32> {
  // expected-error @+1 {{'afir.concat' op all inputs must have the same rank}}
  %0 = afir.concat %arg0, %arg1 {indexing_maps = [#map, #map, #map], concat_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1, 2], vectorized_strides = [16, 4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4xf32>, tensor<3x4x4xf32> -> tensor<5x4x4xf32>
  return %0 : tensor<5x4x4xf32>
}

// ========================================================================
// Verification Error Tests - Cast Operation
// ========================================================================

func.func @test_cast_rank_mismatch(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{'afir.cast' op input and output shapes must match}}
  %0 = afir.cast %arg0 {indexing_maps = [#map, #map1d], outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @test_cast_shape_mismatch(%arg0: tensor<4x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{'afir.cast' op input and output shapes must match}}
  %0 = afir.cast %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [2, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// ========================================================================
// Verification Error Tests - Data Operations
// ========================================================================

func.func @test_load_rank_mismatch(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{'afir.load' op operand and result must have the same rank}}
  %0 = afir.load %arg0 {indexing_maps = [#map, #map1d], outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @test_load_shape_mismatch(%arg0: tensor<4x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{'afir.load' op operand and result must have compatible shapes}}
  %0 = afir.load %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [2, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @test_store_rank_mismatch(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{'afir.store' op operand and result must have the same rank}}
  %0 = afir.store %arg0 {indexing_maps = [#map, #map1d], outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @test_store_shape_mismatch(%arg0: tensor<4x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{'afir.store' op operand and result must have compatible shapes}}
  %0 = afir.store %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [2, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// ========================================================================
// Verification Error Tests - Reduce Operations
// ========================================================================

func.func @test_sum_rank_too_high(%arg0: tensor<4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{'afir.sum' op result rank must be <= input rank for reduce}}
  %0 = afir.sum %arg0 {indexing_maps = [#map1d, #map], axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @test_max_rank_too_high(%arg0: tensor<4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{'afir.max' op result rank must be <= input rank for reduce}}
  %0 = afir.max %arg0 {indexing_maps = [#map1d, #map], axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Verification Error Tests - Broadcast Operations
// ========================================================================

func.func @test_broadcast_rank_too_low(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{'afir.broadcast' op result rank must be >= input rank for broadcast}}
  %0 = afir.broadcast %arg0 {indexing_maps = [#map, #map1d], broadcast_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @test_broadcast_not_broadcastable(%arg0: tensor<4x4xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{'afir.broadcast' op input shape must be broadcastable to result shape}}
  %0 = afir.broadcast %arg0 {indexing_maps = [#map, #map], broadcast_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [8, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
