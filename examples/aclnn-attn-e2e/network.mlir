module {
  func.func private @__aclnn_flash_attention(tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16> attributes {aclnn.layout = "BNSD", aclnn.op = "FlashAttentionScore"}
  func.func @model(%arg0: tensor<1x2x16x8xf16>, %arg1: tensor<1x2x16x8xf16>, %arg2: tensor<1x2x16x8xf16>, %arg3: tensor<1x2x16x16xf16>) -> tensor<2x16x8xf16> {
    %0 = tensor.empty() : tensor<1x2x16x8xf16>
    %cast = tensor.cast %arg0 : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %cast_0 = tensor.cast %arg1 : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %cast_1 = tensor.cast %arg2 : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %cast_2 = tensor.cast %arg3 : tensor<1x2x16x16xf16> to tensor<?x?x?x?xf16>
    %cast_3 = tensor.cast %0 : tensor<1x2x16x8xf16> to tensor<?x?x?x?xf16>
    %1 = call @__aclnn_flash_attention(%cast, %cast_0, %cast_1, %cast_2, %cast_3) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
    %cast_4 = tensor.cast %1 : tensor<?x?x?x?xf16> to tensor<1x2x16x8xf16>
    %collapsed = tensor.collapse_shape %cast_4 [[0, 1], [2], [3]] : tensor<1x2x16x8xf16> into tensor<2x16x8xf16>
    return %collapsed : tensor<2x16x8xf16>
  }
}

