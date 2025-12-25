// RUN: afir-opt --afir-shape-inference %s | FileCheck %s

// Test shape inference pass

// CHECK-LABEL: func.func @test_shape_inference
func.func @test_shape_inference(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: afir.add {{.*}} : tensor<4x4xf32>
  %0 = afir.add %arg0, %arg1 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_chained_shape_inference
func.func @test_chained_shape_inference(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: afir.add {{.*}} : tensor<8x8xf32>
  %0 = afir.add %arg0, %arg1 : tensor<8x8xf32>
  // CHECK: afir.mul {{.*}} : tensor<8x8xf32>
  %1 = afir.mul %0, %arg1 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
