// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize | FileCheck %s --check-prefix=KERNELIZE
// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=target-aware cann-root=%S/Inputs/ascend-schedule-target-tile-cann soc=SyntheticScheduleSoC' --ascend-kernel-split --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s --implicit-check-not=ascendc.mmad --implicit-check-not=ascendc.load_data_with_transpose

// Real 910B1 validation showed f32 batch_matmul with a fused epilogue must
// not enter the mix/cube A2/B2/CO1 path: the generated f32 B1->B2 LoadData
// faults on hardware.

// CHECK: func.func @kernel_
// CHECK: scf.for

// KERNELIZE: linalg.batch_matmul
// KERNELIZE-NOT: ascend.op_role = "cube"
// KERNELIZE-NOT: ascend.kernelize.template_families = ["cube"]
// KERNELIZE: return

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#bias = affine_map<(d0, d1, d2) -> (d2)>

module {
  func.func @f32_batch_matmul_real_npu_fallback(
      %lhs: tensor<?x?x128xf32>,
      %rhs: tensor<?x128x512xf32>,
      %bias_arg: tensor<512xf32>) -> tensor<?x?x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    %batch = tensor.dim %lhs, %c0 : tensor<?x?x128xf32>
    %m = tensor.dim %lhs, %c1 : tensor<?x?x128xf32>
    %init = tensor.empty(%batch, %m) : tensor<?x?x512xf32>
    %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<?x?x512xf32>) -> tensor<?x?x512xf32>
    %matmul = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<?x?x128xf32>, tensor<?x128x512xf32>)
        outs(%filled : tensor<?x?x512xf32>) -> tensor<?x?x512xf32>
    %bias_out = tensor.empty(%batch, %m) : tensor<?x?x512xf32>
    %biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%matmul, %bias_arg : tensor<?x?x512xf32>, tensor<512xf32>)
        outs(%bias_out : tensor<?x?x512xf32>) {
      ^bb0(%in: f32, %bias_value: f32, %out: f32):
        %sum = arith.addf %in, %bias_value : f32
        linalg.yield %sum : f32
    } -> tensor<?x?x512xf32>
    %relu_out = tensor.empty(%batch, %m) : tensor<?x?x512xf32>
    %out = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%biased : tensor<?x?x512xf32>)
        outs(%relu_out : tensor<?x?x512xf32>) {
      ^bb0(%in: f32, %unused: f32):
        %positive = arith.cmpf ugt, %in, %zero : f32
        %selected = arith.select %positive, %in, %zero : f32
        linalg.yield %selected : f32
    } -> tensor<?x?x512xf32>
    return %out : tensor<?x?x512xf32>
  }
}
