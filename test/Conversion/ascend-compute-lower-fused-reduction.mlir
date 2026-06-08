// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=math.erf --implicit-check-not=scf.for

#full = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>
#out = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @gm_fused_gelu_sum_reduction
// CHECK: ascendc.add_l2
// CHECK: ascendc.div_l2
// CHECK: ascendc.erf
// CHECK: ascendc.mul_l2
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
func.func @gm_fused_gelu_sum_reduction(%src: memref<?x?xf32>,
                                       %bias: memref<?xf32>,
                                       %weight: memref<?x?xf32>,
                                       %out: memref<?xf32>) {
  %cst_sqrt2 = arith.constant 1.41421354 : f32
  %cst_one = arith.constant 1.0 : f32
  %cst_half = arith.constant 0.5 : f32
  linalg.generic {
      indexing_maps = [#full, #bias, #full, #out],
      iterator_types = ["parallel", "reduction"]}
      ins(%src, %bias, %weight : memref<?x?xf32>, memref<?xf32>,
                                  memref<?x?xf32>)
      outs(%out : memref<?xf32>) {
    ^bb0(%value: f32, %bias_value: f32, %weight_value: f32, %acc: f32):
      %biased = arith.addf %value, %bias_value : f32
      %scaled = arith.divf %biased, %cst_sqrt2 : f32
      %erf = math.erf %scaled : f32
      %plus = arith.addf %erf, %cst_one : f32
      %half = arith.mulf %plus, %cst_half : f32
      %gelu = arith.mulf %biased, %half : f32
      %weighted = arith.mulf %gelu, %weight_value : f32
      %next = arith.addf %acc, %weighted : f32
      linalg.yield %next : f32
  }
  return
}

#projected_src = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#projected_bias = affine_map<(d0, d1, d2, d3) -> (d3)>
#projected_weight = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#projected_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @gm_projected_fused_gelu_sum_reduction
// CHECK: ascendc.add_l2
// CHECK: ascendc.div_l2
// CHECK: ascendc.erf
// CHECK: ascendc.mul_l2
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK: ascendc.data_copy_l2 {{.*}} : !ascendc.global_tensor<*xf32>, !ascendc.local_tensor<*xf32>, index
func.func @gm_projected_fused_gelu_sum_reduction(
    %src: memref<1x?x256xf32>,
    %bias: memref<256xf32>,
    %weight: memref<64x256xf32>,
    %out: memref<1x?x64xf32>) {
  %cst_sqrt2 = arith.constant 1.41421354 : f32
  %cst_one = arith.constant 1.0 : f32
  %cst_half = arith.constant 0.5 : f32
  linalg.generic {
      indexing_maps = [#projected_src, #projected_bias, #projected_weight,
                       #projected_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%src, %bias, %weight : memref<1x?x256xf32>, memref<256xf32>,
                                  memref<64x256xf32>)
      outs(%out : memref<1x?x64xf32>) {
    ^bb0(%value: f32, %bias_value: f32, %weight_value: f32, %acc: f32):
      %biased = arith.addf %value, %bias_value : f32
      %scaled = arith.divf %biased, %cst_sqrt2 : f32
      %erf = math.erf %scaled : f32
      %plus = arith.addf %erf, %cst_one : f32
      %half = arith.mulf %plus, %cst_half : f32
      %gelu = arith.mulf %biased, %half : f32
      %weighted = arith.mulf %gelu, %weight_value : f32
      %next = arith.addf %acc, %weighted : f32
      linalg.yield %next : f32
  }
  return
}
