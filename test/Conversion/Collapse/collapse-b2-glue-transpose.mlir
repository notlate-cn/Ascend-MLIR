// RUN: afir-opt --auto-fuse-tile-fuse %s | FileCheck %s
//
// Regression: a transpose (class B2) whose input is a tensor.collapse_shape view
// of a func arg — the glue --linalg-fold-unit-extent-dims inserts for a rank-5
// transpose with a unit dim.  hasAnyB2 must look THROUGH the collapse_shape to the
// boundary input; otherwise the B2 guard is skipped, collapse runs on the
// non-collapsible transpose, and emit crashes in ExtractSliceOp::inferResultType
// (operand rank != indexing-map results).  Correct behavior == bare transpose:
// noCollapse, generic keeps reading the rank-4 collapsed value, sliced at rank-4.
//
// CHECK-LABEL: func.func @t5u__v0
// The only collapse_shape is the pre-existing input glue (no NEW collapse added):
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4]] : tensor<1x8x2x3x64xf32> into tensor<8x2x3x64xf32>
// The input is sliced from the rank-4 collapsed value with 4 offsets/sizes:
// CHECK: tensor.extract_slice %collapsed[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : tensor<8x2x3x64xf32> to tensor<?x?x?x?xf32>
// CHECK: linalg.generic
func.func @t5u(%a: tensor<1x8x2x3x64xf32>, %i: tensor<3x8x2x1x64xf32>) -> tensor<3x8x2x1x64xf32> {
  %collapsed = tensor.collapse_shape %a [[0, 1], [2], [3], [4]] : tensor<1x8x2x3x64xf32> into tensor<8x2x3x64xf32>
  %0 = tensor.empty() : tensor<3x8x2x64xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%collapsed : tensor<8x2x3x64xf32>) outs(%0 : tensor<3x8x2x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x8x2x64xf32>
  %expanded = tensor.expand_shape %1 [[0], [1], [2, 3], [4]] output_shape [3, 8, 2, 1, 64] : tensor<3x8x2x64xf32> into tensor<3x8x2x1x64xf32>
  return %expanded : tensor<3x8x2x1x64xf32>
}
