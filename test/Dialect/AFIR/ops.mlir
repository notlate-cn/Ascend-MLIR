// RUN: afir-opt %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1d = affine_map<(d0) -> (d0)>
#map2d = affine_map<(d0, d1) -> (d0, d1)>
#map3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// ========================================================================
// Basic/Math Operations - Unary
// ========================================================================

// CHECK-LABEL: func.func @test_abs
func.func @test_abs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.abs %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_exp
func.func @test_exp(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.exp %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_ln
func.func @test_ln(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.ln %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_sqrt
func.func @test_sqrt(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.sqrt %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_rsqrt
func.func @test_rsqrt(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.rsqrt %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_reciprocal
func.func @test_reciprocal(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.reciprocal %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_erf
func.func @test_erf(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.erf %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_tanh
func.func @test_tanh(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.tanh %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_relu
func.func @test_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.relu %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_neg
func.func @test_neg(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.neg %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_sigmoid
func.func @test_sigmoid(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.sigmoid %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_logical_not
func.func @test_logical_not(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = afir.logical_not %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xi32> -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// CHECK-LABEL: func.func @test_isnan
func.func @test_isnan(%arg0: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.isnan %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_isfinite
func.func @test_isfinite(%arg0: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.isfinite %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_leaky_relu
func.func @test_leaky_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.leaky_relu %arg0 {indexing_maps = [#map, #map], negative_slop = 0.01 : f32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Basic/Math Operations - Binary
// ========================================================================

// CHECK-LABEL: func.func @test_add
func.func @test_add(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_sub
func.func @test_sub(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.sub %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_mul
func.func @test_mul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.mul %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_div
func.func @test_div(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.div %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_minimum
func.func @test_minimum(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.minimum %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_maximum
func.func @test_maximum(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.maximum %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_truediv
func.func @test_truediv(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.truediv %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_pow
func.func @test_pow(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.pow %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_bitwise_and
func.func @test_bitwise_and(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = afir.bitwise_and %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// CHECK-LABEL: func.func @test_floor_div
func.func @test_floor_div(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = afir.floor_div %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// CHECK-LABEL: func.func @test_gelu
func.func @test_gelu(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.gelu %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_sign
func.func @test_sign(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.sign %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_logical_or
func.func @test_logical_or(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xui8> {
  %0 = afir.logical_or %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_logical_and
func.func @test_logical_and(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xui8> {
  %0 = afir.logical_and %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_ge
func.func @test_ge(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.ge %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_eq
func.func @test_eq(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.eq %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_ne
func.func @test_ne(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.ne %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_gt
func.func @test_gt(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.gt %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_le
func.func @test_le(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.le %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// CHECK-LABEL: func.func @test_lt
func.func @test_lt(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xui8> {
  %0 = afir.lt %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xui8>
  return %0 : tensor<4x4xui8>
}

// ========================================================================
// Basic/Math Operations - Ternary
// ========================================================================

// CHECK-LABEL: func.func @test_clip_by_value
func.func @test_clip_by_value(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.clip_by_value %arg0, %arg1, %arg2 {indexing_maps = [#map, #map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Basic/Cast Operations
// ========================================================================

// CHECK-LABEL: func.func @test_cast
func.func @test_cast(%arg0: tensor<4x4xi32>) -> tensor<4x4xf32> {
  %0 = afir.cast %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xi32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Basic/Broadcast Operations
// ========================================================================

// CHECK-LABEL: func.func @test_broadcast
func.func @test_broadcast(%arg0: tensor<4xf32>) -> tensor<4x4xf32> {
  %0 = afir.broadcast %arg0 {indexing_maps = [#map1d, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Basic/Condition Operations
// ========================================================================

// CHECK-LABEL: func.func @test_select
func.func @test_select(%arg0: tensor<4x4xui8>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.select %arg0, %arg1, %arg2 {indexing_maps = [#map, #map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xui8>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_where
func.func @test_where(%arg0: tensor<4x4xui8>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.where %arg0, %arg1, %arg2 {indexing_maps = [#map, #map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xui8>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Basic/Data Operations
// ========================================================================

// CHECK-LABEL: func.func @test_scalar
func.func @test_scalar() -> tensor<f32> {
  %0 = afir.scalar {value = 1.0 : f32, indexing_maps = [], outputs = [#afir.asc_tensor<tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_index_expr
func.func @test_index_expr() -> tensor<i32> {
  %0 = afir.index_expr {indexing_maps = [], outputs = [#afir.asc_tensor<tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @test_load
func.func @test_load(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.load %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_store
func.func @test_store(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.store %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Adv/Reduce Operations
// ========================================================================

// CHECK-LABEL: func.func @test_max_reduce
func.func @test_max_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.max %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_min_reduce
func.func @test_min_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.min %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_sum_reduce
func.func @test_sum_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.sum %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_mean_reduce
func.func @test_mean_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.mean %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_prod_reduce
func.func @test_prod_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.prod %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_any_reduce
func.func @test_any_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.any %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_all_reduce
func.func @test_all_reduce(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.all %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// ========================================================================
// Adv/Concat Operations
// ========================================================================

// CHECK-LABEL: func.func @test_concat
func.func @test_concat(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<9x4xf32> {
  %0 = afir.concat %arg0, %arg1, %arg2 {indexing_maps = [#map2d, #map2d, #map2d, #map2d], concat_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4xf32>, tensor<3x4xf32>, tensor<4x4xf32> -> tensor<9x4xf32>
  return %0 : tensor<9x4xf32>
}

// ========================================================================
// Additional Test Cases - Different Data Types and Shapes
// ========================================================================

// CHECK-LABEL: func.func @test_abs_f16
func.func @test_abs_f16(%arg0: tensor<4x4xf16>) -> tensor<4x4xf16> {
  %0 = afir.abs %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf16> -> tensor<4x4xf16>
  return %0 : tensor<4x4xf16>
}

// CHECK-LABEL: func.func @test_add_i16
func.func @test_add_i16(%arg0: tensor<4x4xi16>, %arg1: tensor<4x4xi16>) -> tensor<4x4xi16> {
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xi16>, tensor<4x4xi16>) -> tensor<4x4xi16>
  return %0 : tensor<4x4xi16>
}

// CHECK-LABEL: func.func @test_unary_1d
func.func @test_unary_1d(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = afir.abs %arg0 {indexing_maps = [#map1d, #map1d], outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<16xf32> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_binary_1d
func.func @test_binary_1d(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map1d, #map1d, #map1d], outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_unary_3d
func.func @test_unary_3d(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = afir.abs %arg0 {indexing_maps = [#map3d, #map3d], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1, 2], vectorized_strides = [32, 8, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// CHECK-LABEL: func.func @test_binary_3d
func.func @test_binary_3d(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map3d, #map3d, #map3d], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1, 2], vectorized_strides = [32, 8, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// CHECK-LABEL: func.func @test_reduce_3d
func.func @test_reduce_3d(%arg0: tensor<2x4x8xf32>) -> tensor<2x4xf32> {
  %0 = afir.sum %arg0 {indexing_maps = [#map3d, #map2d], axis = 2 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4x8xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @test_concat_3d
func.func @test_concat_3d(%arg0: tensor<2x4x8xf32>, %arg1: tensor<3x4x8xf32>) -> tensor<5x4x8xf32> {
  %0 = afir.concat %arg0, %arg1 {indexing_maps = [#map3d, #map3d, #map3d], concat_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1, 2], vectorized_strides = [32, 8, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4x8xf32>, tensor<3x4x8xf32> -> tensor<5x4x8xf32>
  return %0 : tensor<5x4x8xf32>
}

// CHECK-LABEL: func.func @test_cast_f16_to_f32
func.func @test_cast_f16_to_f32(%arg0: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = afir.cast %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf16> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_cast_f32_to_f16
func.func @test_cast_f32_to_f16(%arg0: tensor<4x4xf32>) -> tensor<4x4xf16> {
  %0 = afir.cast %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf16>
  return %0 : tensor<4x4xf16>
}

// CHECK-LABEL: func.func @test_max_reduce_f16
func.func @test_max_reduce_f16(%arg0: tensor<4x4xf16>) -> tensor<4xf16> {
  %0 = afir.max %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf16> -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// CHECK-LABEL: func.func @test_sum_reduce_i32
func.func @test_sum_reduce_i32(%arg0: tensor<4x4xi32>) -> tensor<4xi32> {
  %0 = afir.sum %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xi32> -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// ========================================================================
// Additional Test Cases - Different Outputs Configurations
// ========================================================================

// CHECK-LABEL: func.func @test_abs_output_vector_out
func.func @test_abs_output_vector_out(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.abs %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_add_output_double_buffer
func.func @test_add_output_double_buffer(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = true>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_add_output_reuse
func.func @test_add_output_reuse(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 1, reuse_id = 0, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 1>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_add_output_depth_1
func.func @test_add_output_depth_1(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 1, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_abs_output_different_strides
func.func @test_abs_output_different_strides(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = afir.abs %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [8, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<8x4xf32> -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @test_abs_output_single_axis_vectorization
func.func @test_abs_output_single_axis_vectorization(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.abs %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [1], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_reduce_output_vector_out
func.func @test_reduce_output_vector_out(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  %0 = afir.sum %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_concat_output_vector_out
func.func @test_concat_output_vector_out(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<5x4xf32> {
  %0 = afir.concat %arg0, %arg1 {indexing_maps = [#map2d, #map2d, #map2d], concat_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4xf32>, tensor<3x4xf32> -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func.func @test_cast_output_vector_out
func.func @test_cast_output_vector_out(%arg0: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = afir.cast %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf16> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_select_output_vector_out
func.func @test_select_output_vector_out(%arg0: tensor<4x4xui8>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.select %arg0, %arg1, %arg2 {indexing_maps = [#map, #map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xui8>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_where_output_vector_out
func.func @test_where_output_vector_out(%arg0: tensor<4x4xui8>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.where %arg0, %arg1, %arg2 {indexing_maps = [#map, #map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xui8>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_load_op_vector_out
func.func @test_load_op_vector_out(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.load %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_store_op_vector_out
func.func @test_store_op_vector_out(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = afir.store %arg0 {indexing_maps = [#map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_out, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
