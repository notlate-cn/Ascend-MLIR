// RUN: afir-opt --afir-canonicalize %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test canonicalization pass

// CHECK-LABEL: func.func @test_basic_canonicalize
func.func @test_basic_canonicalize(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: afir.add
  %0 = afir.add %arg0, %arg1 indexing_maps = [#map, #map, #map] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Future canonicalization patterns can be tested here
// Examples:
// - add(x, 0) -> x
// - mul(x, 1) -> x
// - sub(x, x) -> constant 0
