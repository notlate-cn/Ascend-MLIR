// RUN: afir-opt --fuse-transpose-into-elementwise %s | FileCheck %s

// Phase-1 gated "eliminate transpose": a transpose whose sole consumer is a
// pure-elementwise generic, in the f16 rank-2 [1,0] form the on-chip transpose
// template can codegen, is absorbed into the consumer's input indexing map.
// Everything else (f32, multi-use, non-elementwise consumer) is left untouched
// and stays a separate transpose → aclnn.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#red = affine_map<(d0, d1) -> (d0)>

// The swapped (d1,d0) map only exists because the transpose was folded in.
// CHECK-DAG: affine_map<(d0, d1) -> (d1, d0)>

// ---------------------------------------------------------------------------
// f16 [1,0] transpose feeding a relu generic → FOLDED (transpose gone, the
// generic now reads the original operand directly — which is only type-correct
// with the swapped map).
// CHECK-LABEL: func.func @fold_f16_transpose_relu
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<16x32xf16>)
func.func @fold_f16_transpose_relu(%x: tensor<16x32xf16>) -> tensor<32x16xf16> {
  %zero = arith.constant 0.0 : f16
  %t_init = tensor.empty() : tensor<32x16xf16>
  %t = linalg.transpose ins(%x : tensor<16x32xf16>)
                        outs(%t_init : tensor<32x16xf16>) permutation = [1, 0]
  %r_init = tensor.empty() : tensor<32x16xf16>
  %out = linalg.generic {
      indexing_maps = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<32x16xf16>) outs(%r_init : tensor<32x16xf16>) {
  ^bb0(%in: f16, %o: f16):
    %v = arith.maximumf %in, %zero : f16
    linalg.yield %v : f16
  } -> tensor<32x16xf16>
  return %out : tensor<32x16xf16>
}

// ---------------------------------------------------------------------------
// f32: vtranspose can't → NOT folded (transpose stays).
// CHECK-LABEL: func.func @keep_f32_transpose_relu
// CHECK:         linalg.transpose
func.func @keep_f32_transpose_relu(%x: tensor<16x32xf32>) -> tensor<32x16xf32> {
  %zero = arith.constant 0.0 : f32
  %t_init = tensor.empty() : tensor<32x16xf32>
  %t = linalg.transpose ins(%x : tensor<16x32xf32>)
                        outs(%t_init : tensor<32x16xf32>) permutation = [1, 0]
  %r_init = tensor.empty() : tensor<32x16xf32>
  %out = linalg.generic {
      indexing_maps = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<32x16xf32>) outs(%r_init : tensor<32x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %v = arith.maximumf %in, %zero : f32
    linalg.yield %v : f32
  } -> tensor<32x16xf32>
  return %out : tensor<32x16xf32>
}

// ---------------------------------------------------------------------------
// Two consumers of the transpose → NOT folded (sole-use gate).
// CHECK-LABEL: func.func @keep_multi_use_transpose
// CHECK:         linalg.transpose
func.func @keep_multi_use_transpose(%x: tensor<16x32xf16>) -> tensor<32x16xf16> {
  %zero = arith.constant 0.0 : f16
  %t_init = tensor.empty() : tensor<32x16xf16>
  %t = linalg.transpose ins(%x : tensor<16x32xf16>)
                        outs(%t_init : tensor<32x16xf16>) permutation = [1, 0]
  %a_init = tensor.empty() : tensor<32x16xf16>
  %a = linalg.generic {
      indexing_maps = [#id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<32x16xf16>) outs(%a_init : tensor<32x16xf16>) {
  ^bb0(%in: f16, %o: f16):
    %v = arith.maximumf %in, %zero : f16
    linalg.yield %v : f16
  } -> tensor<32x16xf16>
  %b_init = tensor.empty() : tensor<32x16xf16>
  %b = linalg.generic {
      indexing_maps = [#id2, #id2, #id2],
      iterator_types = ["parallel", "parallel"]}
      ins(%t, %a : tensor<32x16xf16>, tensor<32x16xf16>)
      outs(%b_init : tensor<32x16xf16>) {
  ^bb0(%in0: f16, %in1: f16, %o: f16):
    %v = arith.addf %in0, %in1 : f16
    linalg.yield %v : f16
  } -> tensor<32x16xf16>
  return %b : tensor<32x16xf16>
}

// ---------------------------------------------------------------------------
// Consumer has a reduction iterator (not pure-elementwise) → NOT folded.
// CHECK-LABEL: func.func @keep_transpose_into_reduce
// CHECK:         linalg.transpose
func.func @keep_transpose_into_reduce(%x: tensor<16x32xf16>) -> tensor<32xf16> {
  %t_init = tensor.empty() : tensor<32x16xf16>
  %t = linalg.transpose ins(%x : tensor<16x32xf16>)
                        outs(%t_init : tensor<32x16xf16>) permutation = [1, 0]
  %r_init = tensor.empty() : tensor<32xf16>
  %out = linalg.generic {
      indexing_maps = [#id2, #red],
      iterator_types = ["parallel", "reduction"]}
      ins(%t : tensor<32x16xf16>) outs(%r_init : tensor<32xf16>) {
  ^bb0(%in: f16, %o: f16):
    %v = arith.addf %in, %o : f16
    linalg.yield %v : f16
  } -> tensor<32xf16>
  return %out : tensor<32xf16>
}
