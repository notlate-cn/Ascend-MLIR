// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=math.erf --implicit-check-not=ascendc.abs_l2

// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
// math.erf lowers through PyAsc's AscendC math-library Erf op.

func.func @gelu(%input: tensor<4x128xf32>) -> tensor<4x128xf32> {
  %cst_rsqrt2 = arith.constant 0.7071067811865476 : f32
  %cst_half = arith.constant 0.5 : f32
  %cst_one = arith.constant 1.0 : f32

  %out_init = tensor.empty() : tensor<4x128xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x128xf32>)
    outs(%out_init : tensor<4x128xf32>) {
  ^bb0(%x: f32, %out: f32):
    %scaled = arith.mulf %x, %cst_rsqrt2 : f32
    %e = math.erf %scaled : f32
    %p1 = arith.addf %e, %cst_one : f32
    %half_p1 = arith.mulf %p1, %cst_half : f32
    %y = arith.mulf %x, %half_p1 : f32
    linalg.yield %y : f32
  } -> tensor<4x128xf32>

  return %result : tensor<4x128xf32>
}

// CHECK: func.func @gelu
// CHECK: ascendc.erf
// CHECK: return
