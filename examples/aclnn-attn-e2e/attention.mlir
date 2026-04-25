// End-to-end test input for tm_tensor.attention → aclnn direct call.
//
// Shape: B=1, N=2, S=16, D=8  dtype=f16
// Layout: inputs arrive as 4D [B,N,S,D]; torch-mlir typically collapses B*N→BN
// before tm_tensor.attention.  ConvertTmTensorAttentionPass detects the
// collapse_shape and uses the pre-collapse 4D tensors directly for CANN.
//
// RUN: torch-opt --convert-tm-tensor-attention %s \
// RUN:   | afir-opt --aclnn-finalize-decl \
// RUN:   | FileCheck %s
//
// CHECK: aclnn.op = "FlashAttentionScore"
// CHECK: aclnn.layout = "BNSD"
// CHECK: call @__aclnn_flash_attention

module {
  func.func @model(
      %q4d:    tensor<1x2x16x8xf16>,   // [B, N, S, D]
      %k4d:    tensor<1x2x16x8xf16>,
      %v4d:    tensor<1x2x16x8xf16>,
      %mask4d: tensor<1x2x16x16xf16>   // [B, N, S, S] additive mask (all zeros → no mask)
  ) -> tensor<2x16x8xf16> {

    // Collapse [B,N,S,D] → [B*N, S, D] as torch-mlir would.
    %q    = tensor.collapse_shape %q4d    [[0, 1], [2], [3]]
                : tensor<1x2x16x8xf16>  into tensor<2x16x8xf16>
    %k    = tensor.collapse_shape %k4d    [[0, 1], [2], [3]]
                : tensor<1x2x16x8xf16>  into tensor<2x16x8xf16>
    %v    = tensor.collapse_shape %v4d    [[0, 1], [2], [3]]
                : tensor<1x2x16x8xf16>  into tensor<2x16x8xf16>
    %mask = tensor.collapse_shape %mask4d [[0, 1], [2], [3]]
                : tensor<1x2x16x16xf16> into tensor<2x16x16xf16>

    %cst  = arith.constant 0.0 : f16
    %init = tensor.empty() : tensor<2x16x8xf16>
    %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<2x16x8xf16>)
                -> tensor<2x16x8xf16>

    %out = tm_tensor.attention
        ins(%q, %k, %v, %mask :
            tensor<2x16x8xf16>, tensor<2x16x8xf16>,
            tensor<2x16x8xf16>, tensor<2x16x16xf16>)
        outs(%fill : tensor<2x16x8xf16>) -> tensor<2x16x8xf16>

    return %out : tensor<2x16x8xf16>
  }
}