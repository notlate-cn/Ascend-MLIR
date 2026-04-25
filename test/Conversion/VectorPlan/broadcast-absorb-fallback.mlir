// RUN: afir-opt --vector-plan-broadcast-absorb %s | FileCheck %s

// CHECK-LABEL: func.func @fallback_no_generic_consumer
func.func @fallback_no_generic_consumer(%x: tensor<4x8xf16>) -> tensor<4x3x8xf16> {
  %init = tensor.empty() : tensor<4x3x8xf16>
  %b = linalg.broadcast ins(%x : tensor<4x8xf16>)
                        outs(%init : tensor<4x3x8xf16>)
                        dimensions = [1]
  return %b : tensor<4x3x8xf16>
}
// CHECK: linalg.broadcast
