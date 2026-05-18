// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// Multi-reduce-axes, adjacent (no parallel axis between r1 and r2).
//   out[a] = sum_{r1, r2}( x[a, r1, r2] )
//
// Two contiguous reduction iters with same role → the Collapse pass merges
// them into one R axis of extent R1*R2.  After collapse the kernel is a 2D
// AR-pattern reduce, same path as a single-R reduce.  Verifies:
//   - operand `tensor<8x16x32xf32>` → collapsed to `tensor<8x512xf32>`
//   - iter space goes from 3D `[parallel, reduction, reduction]` to 2D
//     `[parallel, reduction]`
//   - picker lands on FullLoad (R=512 fits whole on-chip)
//
// Non-adjacent multi-R (parallel iter between two reduction iters, e.g.
// `out[a] = sum_{r1,r2} x[r1, a, r2]`) is **NOT yet supported** —
// ComputeConversion assumes operand layout is parallel-prefix +
// reduction-suffix, which the interleaved memory order violates.  See
// `project_af_scheduler_port` memory for the gap notes.

// CHECK: func.func @multi_r_adj__v0(
// CHECK-SAME: afir.reduce_template = "FullLoad"
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2]] : tensor<8x16x32xf32> into tensor<8x512xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]

func.func @multi_r_adj(%x: tensor<8x16x32xf32>,
                        %init: tensor<8xf32>) -> tensor<8xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(a, r1, r2) -> (a, r1, r2)>,
                     affine_map<(a, r1, r2) -> (a)>],
    iterator_types = ["parallel", "reduction", "reduction"]}
    ins(%x : tensor<8x16x32xf32>)
    outs(%init : tensor<8xf32>) {
  ^bb0(%v: f32, %acc: f32):
    %t = arith.addf %v, %acc : f32
    linalg.yield %t : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}
