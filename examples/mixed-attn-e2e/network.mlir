// Mixed network for the v1 end-to-end demo.
//
// Pipeline:
//   q' = q * scale + bias              (kernel_group0: AscendC elementwise via auto-fuse-codegen)
//   out = softmax(q'·k^T/√d + mask)·v  (aclnn FlashAttentionScore, dispatched host-mode CPU reference)
//
// kernel_group1 is intentionally absent — a "second AscendC launch after FA in the
// same process" exhibits a >20-minute camodel hang (single launch standalone runs
// in <1s). One AscendC + one aclnn is enough to exercise the mixed-compilation path.

module {
  func.func private @kernel_group0(tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>,
                                    tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>)
      -> tensor<1x1x2x8xf16>

  func.func private @__aclnn_flash_attention(
      tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
      tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
      attributes {aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"}

  func.func @model(
      %q: tensor<1x1x2x8xf16>, %scale: tensor<1x1x2x8xf16>, %bias: tensor<1x1x2x8xf16>,
      %k: tensor<1x1x2x8xf16>, %v: tensor<1x1x2x8xf16>,
      %mask: tensor<1x1x2x2xf16>, %init_pre: tensor<1x1x2x8xf16>,
      %init_fa: tensor<1x1x2x8xf16>) -> tensor<1x1x2x8xf16> {
    %qprime = call @kernel_group0(%q, %scale, %bias, %init_pre)
        : (tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>, tensor<1x1x2x8xf16>,
           tensor<1x1x2x8xf16>) -> tensor<1x1x2x8xf16>
    %qc  = tensor.cast %qprime    : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %kc  = tensor.cast %k         : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %vc  = tensor.cast %v         : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %mc  = tensor.cast %mask      : tensor<1x1x2x2xf16> to tensor<?x?x?x?xf16>
    %ic  = tensor.cast %init_fa   : tensor<1x1x2x8xf16> to tensor<?x?x?x?xf16>
    %fa  = call @__aclnn_flash_attention(%qc, %kc, %vc, %mc, %ic)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
           tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
    %fac = tensor.cast %fa : tensor<?x?x?x?xf16> to tensor<1x1x2x8xf16>
    return %fac : tensor<1x1x2x8xf16>
  }
}
