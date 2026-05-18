// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s --check-prefix=TILE
// RUN: afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s --check-prefix=ASCENDC
//
// Leading-axis reduce (RA pattern + FullLoad): out[d1] = sum_{d0} x[d0,d1].
// Reduce iter (d0) sits BEFORE the parallel iter (d1) in the input's
// indexing map; combined with x's [d0,d1] physical layout that makes the
// operand [R, A] = RA layout.  R=8 fits whole on-chip, no row-loop degrade
// → costEstimate picks FullLoad.  ComputeConversion's AR/RA detection is
// template-agnostic; FullLoad + RA must still emit reduce_sum_2d_l2 with
// the RA layout attribute (= 1 in AscendC_ReduceLayoutAttr).

// Picker: FullLoad wins, no second variant — RA isn't a separate template.
// TILE: func.func @leading_reduce__v0(
// TILE-SAME: afir.reduce_template = "FullLoad"
// TILE-NOT: func.func @leading_reduce__v1

// Lowered kernel: single reduce_sum_2d_l2 with layout = 1 (RA).
// ASCENDC: func.func @leading_reduce__v0(
// ASCENDC-SAME: afir.reduce_template = "FullLoad"
// ASCENDC: ascendc.reduce_sum_2d_l2
// ASCENDC-SAME: layout = 1 : i32

func.func @leading_reduce(%x: tensor<8x1024xf32>,
                          %init: tensor<1024xf32>) -> tensor<1024xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%x : tensor<8x1024xf32>)
    outs(%init : tensor<1024xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %t = arith.addf %a, %acc : f32
    linalg.yield %t : f32
  } -> tensor<1024xf32>
  return %r : tensor<1024xf32>
}
