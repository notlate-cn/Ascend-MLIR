// RUN: torch-opt --convert-tm-tensor-attention %s | FileCheck %s

// Verifies that tm_tensor.attention is lowered to func.call @__aclnn_flash_attention.
// Q/K/V/mask come from tensor.collapse_shape on 4D [B,N,S,D] sources;
// the pass bypasses the collapse and calls aclnn with the original 4D values.

// -----
// Static shapes: [B=1, N=2, S=8, D=4] → collapse → [2, 8, 4]

// CHECK: func.func private @__aclnn_flash_attention
// CHECK-SAME: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
// CHECK-SAME: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
// CHECK-SAME: -> tensor<?x?x?x?xf32>
// CHECK-SAME: aclnn.kind = "flash_attention"

func.func @attention_static(
    %q4d:    tensor<1x2x8x4xf32>,
    %k4d:    tensor<1x2x8x4xf32>,
    %v4d:    tensor<1x2x8x4xf32>,
    %mask4d: tensor<1x2x8x8xf32>
) -> tensor<2x8x4xf32> {
  %q    = tensor.collapse_shape %q4d    [[0, 1], [2], [3]] : tensor<1x2x8x4xf32> into tensor<2x8x4xf32>
  %k    = tensor.collapse_shape %k4d    [[0, 1], [2], [3]] : tensor<1x2x8x4xf32> into tensor<2x8x4xf32>
  %v    = tensor.collapse_shape %v4d    [[0, 1], [2], [3]] : tensor<1x2x8x4xf32> into tensor<2x8x4xf32>
  %mask = tensor.collapse_shape %mask4d [[0, 1], [2], [3]] : tensor<1x2x8x8xf32> into tensor<2x8x8xf32>
  %cst  = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2x8x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2x8x4xf32>) -> tensor<2x8x4xf32>

  %out = tm_tensor.attention
      ins(%q, %k, %v, %mask : tensor<2x8x4xf32>, tensor<2x8x4xf32>, tensor<2x8x4xf32>, tensor<2x8x8xf32>)
      outs(%fill : tensor<2x8x4xf32>) -> tensor<2x8x4xf32>

  return %out : tensor<2x8x4xf32>
}

// The func.call should use the 4D pre-collapse values (cast to dynamic).
// CHECK-LABEL: func.func @attention_static
// CHECK:         %[[Q4D:.+]] = tensor.cast {{.*}} : tensor<1x2x8x4xf32> to tensor<?x?x?x?xf32>
// CHECK:         %[[K4D:.+]] = tensor.cast {{.*}} : tensor<1x2x8x4xf32> to tensor<?x?x?x?xf32>
// CHECK:         %[[V4D:.+]] = tensor.cast {{.*}} : tensor<1x2x8x4xf32> to tensor<?x?x?x?xf32>
// CHECK:         %[[M4D:.+]] = tensor.cast {{.*}} : tensor<1x2x8x8xf32> to tensor<?x?x?x?xf32>
// CHECK:         %[[RES:.+]] = call @__aclnn_flash_attention(%[[Q4D]], %[[K4D]], %[[V4D]], %[[M4D]], {{.*}})
// CHECK:         tensor.cast %[[RES]] : tensor<?x?x?x?xf32> to tensor<1x2x8x4xf32>
// CHECK-NOT:     tm_tensor.attention

// -----
// Dynamic shapes: [B=?, N=2, S=?, D=4] → collapse → [?, ?, 4]

func.func @attention_dynamic(
    %q4d:    tensor<?x2x?x4xf32>,
    %k4d:    tensor<?x2x?x4xf32>,
    %v4d:    tensor<?x2x?x4xf32>,
    %mask4d: tensor<?x2x?x?xf32>
) -> tensor<?x?x4xf32> {
  %q    = tensor.collapse_shape %q4d    [[0, 1], [2], [3]] : tensor<?x2x?x4xf32> into tensor<?x?x4xf32>
  %k    = tensor.collapse_shape %k4d    [[0, 1], [2], [3]] : tensor<?x2x?x4xf32> into tensor<?x?x4xf32>
  %v    = tensor.collapse_shape %v4d    [[0, 1], [2], [3]] : tensor<?x2x?x4xf32> into tensor<?x?x4xf32>
  %mask = tensor.collapse_shape %mask4d [[0, 1], [2], [3]] : tensor<?x2x?x?xf32> into tensor<?x?x?xf32>
  %cst  = arith.constant 0.0 : f32
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %d0   = tensor.dim %q, %c0 : tensor<?x?x4xf32>
  %d1   = tensor.dim %q, %c1 : tensor<?x?x4xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x4xf32>) -> tensor<?x?x4xf32>

  %out = tm_tensor.attention
      ins(%q, %k, %v, %mask : tensor<?x?x4xf32>, tensor<?x?x4xf32>, tensor<?x?x4xf32>, tensor<?x?x?xf32>)
      outs(%fill : tensor<?x?x4xf32>) -> tensor<?x?x4xf32>

  return %out : tensor<?x?x4xf32>
}

// Dynamic case: inputs are already dynamic 4D, no extra cast needed.
// CHECK-LABEL: func.func @attention_dynamic
// CHECK:         call @__aclnn_flash_attention
// CHECK-NOT:     tm_tensor.attention