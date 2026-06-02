// Integration: the network phase-1 pass order (network_runner.py) — fold-unit /
// canonicalize / fuse-transpose-into-elementwise / canonicalize / group-analysis.
// Proves the gated fold runs in the network path: an f16 [1,0] transpose feeding
// an elementwise generic is absorbed into a single vector group (no standalone
// transpose → no aclnn Transpose kernel); an f32 transpose is left untouched and
// survives as its own op (group-outline routes it to aclnn — unchanged).
//
// RUN: afir-opt --linalg-fold-unit-extent-dims --canonicalize \
// RUN:   --fuse-transpose-into-elementwise --canonicalize \
// RUN:   --auto-fuse-group-analysis=disable-cube-fusion=true %s | FileCheck %s

#id2 = affine_map<(d0, d1) -> (d0, d1)>

// f16: folded → one elementwise group reading the original operand, no transpose.
// CHECK-LABEL: func.func @fold_f16
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0 : tensor<16x32xf16>)
// CHECK-SAME:      auto_fuse.group_id
func.func @fold_f16(%x: tensor<16x32xf16>) -> tensor<32x16xf16> {
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

// f32: gate rejects → transpose survives (→ aclnn, unchanged).
// CHECK-LABEL: func.func @keep_f32
// CHECK:         linalg.transpose
func.func @keep_f32(%x: tensor<16x32xf32>) -> tensor<32x16xf32> {
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
