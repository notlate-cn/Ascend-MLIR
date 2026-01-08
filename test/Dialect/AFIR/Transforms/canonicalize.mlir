// RUN: afir-opt --afir-canonicalize %s | FileCheck %s

// Test canonicalization pass

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @test_basic_canonicalize
func.func @test_basic_canonicalize(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: afir.add
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = VECTOR_IN, position_id = 0, depth = 0, is_double_buffer = false>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Future canonicalization patterns can be tested here
// Examples:
// - add(x, 0) -> x
// - mul(x, 1) -> x
// - sub(x, x) -> constant 0
