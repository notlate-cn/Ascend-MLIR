// 3-segment kg1-hang repro:
//   kg0 (static elementwise) → aclnn FA (host-mode CPU ref) → kg1 (dynamic elementwise add).
// Exists to localize where the second AscendC launch hangs in the same process.
//
// Shapes intentionally kept tiny (1×1×2×8 fp16) so the codegen path matches
// the v1 mixed-attn-e2e demo exactly. kg1 is dynamic-shape to verify the
// SymExpr path still hangs (or doesn't) the same way as static.

module {
  func.func private @kernel_group0(tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>,
                                    tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>)
      -> tensor<1x1x2x8xf16>

  func.func private @__aclnn_flash_attention(
      tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
      tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
      attributes {aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"}

  func.func private @kernel_group1(tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
                                    tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>

  func.func @model(
      %q: tensor<1x1x2x8xf16>, %scale: tensor<1x1x2x8xf16>, %bias: tensor<1x1x2x8xf16>,
      %k: tensor<1x1x2x8xf16>, %v: tensor<1x1x2x8xf16>,
      %mask: tensor<1x1x2x2xf16>,
      %init_pre: tensor<1x1x2x8xf16>, %init_fa: tensor<1x1x2x8xf16>,
      %kg1_bias: tensor<1x1x2x8xf16>, %init_post: tensor<1x1x2x8xf16>)
      -> tensor<1x1x2x8xf16> {
    %qprime = call @kernel_group0(%q, %scale, %bias, %init_pre)
        : (tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>,
           tensor<1x1x2x8xf16>) -> tensor<1x1x2x8xf16>
    %qc  = tensor.cast %qprime  : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %kc  = tensor.cast %k       : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %vc  = tensor.cast %v       : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %mc  = tensor.cast %mask    : tensor<1x1x2x2xf16> to tensor<?x?x?x?xf16>
    %ic  = tensor.cast %init_fa : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %fa  = call @__aclnn_flash_attention(%qc, %kc, %vc, %mc, %ic)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
           tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
    %bc  = tensor.cast %kg1_bias  : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %ipc = tensor.cast %init_post : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %res = call @kernel_group1(%fa, %bc, %ipc)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)
        -> tensor<?x?x?x?xf16>
    %resc = tensor.cast %res : tensor<?x?x?x?xf16> to tensor<1x1x2x8xf16>
    return %resc : tensor<1x1x2x8xf16>
  }
}
