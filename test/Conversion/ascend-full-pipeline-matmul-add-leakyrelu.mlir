// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s --implicit-check-not=linalg.

// CHECK-LABEL: func.func @matmul_add_leakyrelu
// CHECK-SAME: ascendc.kernel_kind = "mix"
// CHECK: ascendc.mmad
// CHECK: ascendc.data_copy_co12dst
// CHECK: ascendc.add_l2
// CHECK: ascendc.mul_l2
// CHECK: ascendc.max_l2

#identity = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

module {
  func.func @matmul_add_leakyrelu(
      %lhs: tensor<?x?xf16>,
      %rhs: tensor<?x?xf16>,
      %bias_arg: tensor<?xf32>,
      %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %matmul = linalg.matmul
        ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %out, %c0 : tensor<?x?xf32>
    %n = tensor.dim %out, %c1 : tensor<?x?xf32>
    %bias_out = tensor.empty(%m, %n) : tensor<?x?xf32>
    %biased = linalg.generic {
        indexing_maps = [#identity, #bias, #identity],
        iterator_types = ["parallel", "parallel"]
      } ins(%matmul, %bias_arg : tensor<?x?xf32>, tensor<?xf32>)
        outs(%bias_out : tensor<?x?xf32>) {
      ^bb0(%x: f32, %bias: f32, %acc: f32):
        %sum = arith.addf %x, %bias : f32
        linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    %alpha = arith.constant 1.000000e-03 : f32
    %relu_out = tensor.empty(%m, %n) : tensor<?x?xf32>
    %result = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel", "parallel"]
      } ins(%biased : tensor<?x?xf32>)
        outs(%relu_out : tensor<?x?xf32>) {
      ^bb0(%x: f32, %acc: f32):
        %scaled = arith.mulf %x, %alpha : f32
        %max = arith.maximumf %x, %scaled : f32
        linalg.yield %max : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}
