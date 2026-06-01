// RUN: not afir-opt %s --auto-fuse-codegen 2>&1 | FileCheck %s

// AscendC::Transpose lowers to a single 16x16 `vtranspose`, which only handles
// 16-bit data (half/int16/uint16).  A transpose that reaches the on-chip
// codegen path (here the multi-consumer "preserve" template) with an f32
// element type must fail loudly instead of silently emitting a wrong
// ascendc.transpose.  The 16-bit, rank-2 [1,0] case still lowers (see
// examples/transpose-preserve-e2e).

// CHECK: error: LinalgToAscendC: AscendC::Transpose only supports a rank-2 [1,0] transpose of 16-bit data

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @transpose_f32(%x: tensor<32x32xf32>)
      -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %zero = arith.constant 0.0 : f32
    %two = arith.constant 2.0 : f32
    %t_init = tensor.empty() : tensor<32x32xf32>
    %t = linalg.transpose ins(%x : tensor<32x32xf32>)
                          outs(%t_init : tensor<32x32xf32>) permutation = [1, 0]
    %a_init = tensor.empty() : tensor<32x32xf32>
    %a = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x32xf32>) outs(%a_init : tensor<32x32xf32>) {
    ^bb0(%v: f32, %o: f32):
      %m = arith.maximumf %v, %zero : f32
      linalg.yield %m : f32
    } -> tensor<32x32xf32>
    %b_init = tensor.empty() : tensor<32x32xf32>
    %b = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x32xf32>) outs(%b_init : tensor<32x32xf32>) {
    ^bb0(%v: f32, %o: f32):
      %m = arith.mulf %v, %two : f32
      linalg.yield %m : f32
    } -> tensor<32x32xf32>
    return %a, %b : tensor<32x32xf32>, tensor<32x32xf32>
  }
}
