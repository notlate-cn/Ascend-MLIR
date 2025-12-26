// RUN: afir-opt --convert-afir-to-ascir %s | FileCheck %s

// Test AFIR to ASC-IR conversion for add operation

// Test case 1: Basic tensor addition
// CHECK-LABEL: func.func @convert_add_basic
func.func @convert_add_basic(%arg0: !ascendc.local_tensor<4x4xf32>, %arg1: !ascendc.local_tensor<4x4xf32>) -> !ascendc.local_tensor<4x4xf32> {
  // CHECK: ascir.add %arg0, %arg1 : ascendc.local_tensor<4x4xf32>
  %0 = afir.add %arg0, %arg1 : !ascendc.local_tensor<4x4xf32>
  return %0 : !ascendc.local_tensor<4x4xf32>
}
