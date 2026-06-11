// RUN: not afir-opt %s --auto-fuse-codegen 2>&1 | FileCheck %s

// The on-chip transpose lowers to codegen::AfirConfusionTranspose2D
// (TransDataTo5HD-based), which handles f16/f32 rank-2 [1,0] at any size. A
// transpose reaching the on-chip codegen path (here the multi-consumer
// "preserve" template) with an unsupported element type (bf16: 16-bit but not
// f16/f32) must fail loudly instead of silently emitting a wrong transpose.
// The f16/f32 rank-2 [1,0] cases still lower (see transpose-preserve-e2e and
// the f32 transpose+relu validated on sim).

// CHECK: error: LinalgToAscendC: on-chip transpose supports only a rank-2 [1,0] transpose of f16/f32 data

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @transpose_bf16(%x: tensor<32x32xbf16>)
      -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %zero = arith.constant 0.0 : bf16
    %two = arith.constant 2.0 : bf16
    %t_init = tensor.empty() : tensor<32x32xbf16>
    %t = linalg.transpose ins(%x : tensor<32x32xbf16>)
                          outs(%t_init : tensor<32x32xbf16>) permutation = [1, 0]
    %a_init = tensor.empty() : tensor<32x32xbf16>
    %a = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x32xbf16>) outs(%a_init : tensor<32x32xbf16>) {
    ^bb0(%v: bf16, %o: bf16):
      %m = arith.maximumf %v, %zero : bf16
      linalg.yield %m : bf16
    } -> tensor<32x32xbf16>
    %b_init = tensor.empty() : tensor<32x32xbf16>
    %b = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x32xbf16>) outs(%b_init : tensor<32x32xbf16>) {
    ^bb0(%v: bf16, %o: bf16):
      %m = arith.mulf %v, %two : bf16
      linalg.yield %m : bf16
    } -> tensor<32x32xbf16>
    return %a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
