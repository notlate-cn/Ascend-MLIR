// RUN: afir-opt --convert-afir-to-ascir %s | FileCheck %s

// Test AFIR to ASC-IR conversion for add operation

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1d = affine_map<(d0) -> (d0)>
#map3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// Test case 1: Basic tensor addition
// CHECK-LABEL: func.func @convert_add_basic
func.func @convert_add_basic(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: %0 = ascendc.tbuf : <veccalc>
  // CHECK: %1 = ascendc.tbuf.get_tensor %0 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<4x4xf32>
  // CHECK: ascendc.add_l3 %1, %arg0, %arg1 : !ascendc.local_tensor<4x4xf32>, !ascendc.local_tensor<4x4xf32>, !ascendc.local_tensor<4x4xf32>
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id=2, position = <vector_in, depth=2, is_double_buffer=true>, position_id=2>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Conversion Test Cases - Different Data Types and Shapes
// ========================================================================

// Test case 2: Addition with f16 data type
// CHECK-LABEL: func.func @convert_add_f16
func.func @convert_add_f16(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf16> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf16>, tensor<4x4xf16>) -> tensor<4x4xf16>
  return %0 : tensor<4x4xf16>
}

// Test case 3: Addition with i32 data type
// CHECK-LABEL: func.func @convert_add_i32
func.func @convert_add_i32(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// Test case 4: Addition with 1D tensor
// CHECK-LABEL: func.func @convert_add_1d
func.func @convert_add_1d(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map1d, #map1d, #map1d], outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// Test case 5: Addition with 3D tensor
// CHECK-LABEL: func.func @convert_add_3d
func.func @convert_add_3d(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map3d, #map3d, #map3d], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1, 2], vectorized_strides = [32, 8, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// ========================================================================
// Conversion Test Cases - Different Operations
// ========================================================================

// Test case 6: Mul operation
// CHECK-LABEL: func.func @convert_mul
func.func @convert_mul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ascendc.mul_l3
  %0 = afir.mul %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test case 7: Sub operation
// CHECK-LABEL: func.func @convert_sub
func.func @convert_sub(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ascendc.sub_l3
  %0 = afir.sub %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test case 8: Div operation
// CHECK-LABEL: func.func @convert_div
func.func @convert_div(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ascendc.div_l3
  %0 = afir.div %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// ========================================================================
// Conversion Test Cases - Edge Cases
// ========================================================================

// Test case 9: Small tensor (2x2)
// CHECK-LABEL: func.func @convert_small_tensor
func.func @convert_small_tensor(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [2, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// Test case 10: Large tensor (16x16)
// CHECK-LABEL: func.func @convert_large_tensor
func.func @convert_large_tensor(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [16, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// Test case 11: Non-square tensor (8x4)
// CHECK-LABEL: func.func @convert_non_square_tensor
func.func @convert_non_square_tensor(%arg0: tensor<8x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// Test case 12: Multiple operations in sequence
// CHECK-LABEL: func.func @convert_multiple_ops
func.func @convert_multiple_ops(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ascendc.add_l3
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: ascendc.mul_l3
  %1 = afir.mul %0, %arg2 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 1, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 1>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
