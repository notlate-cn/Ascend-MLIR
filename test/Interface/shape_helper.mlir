// RUN: afir-opt --afir-shape-inference %s | FileCheck %s

// Test ShapeHelperOpInterface

// CHECK-LABEL: func.func @test_binary_shape_helper
func.func @test_binary_shape_helper(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // Shape helper should infer output shape from inputs
  // CHECK: afir.add {{.*}} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %0 = afir.add %arg0, %arg1 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @test_shape_helper_chain
func.func @test_shape_helper_chain(%a: tensor<5x5xf32>, %b: tensor<5x5xf32>, %c: tensor<5x5xf32>) -> tensor<5x5xf32> {
  // CHECK: afir.sub {{.*}} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %0 = afir.sub %a, %b : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: afir.mul {{.*}} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %1 = afir.mul %0, %0 : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  // CHECK: afir.div {{.*}} : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  %2 = afir.div %1, %c : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>
  return %2 : tensor<5x5xf32>
}
