// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// Non-contiguous ("displaced") multi-reduce-axes:
//   out[a] = sum_{r1, r2}( x[r1, a, r2] )
//
// Iterator types `[reduction, parallel, reduction]` — the parallel axis
// between the two reductions stops Collapse from merging them.  FullLoad sims
// wrong on this shape (ComputeConversion's `redDim = first reduction iter dim`
// heuristic mis-picks the RA layout when the buffer is actually AR), so the
// picker rejects FullLoad and lands on the new peel-outer-R draft: the
// outermost displaced R becomes a step=1 scf.for around a rank-reduced 2-D
// inner linalg.generic.  Verifies:
//   - reduce_template = "Common" (not "FullLoad")
//   - an `scf.for ... step %c1` with bounds 0..R1 is emitted around the
//     reduce
//   - the inner generic has rank-2 iter_types [parallel, reduction] (the
//     displaced R has been peeled away)
//   - one of the inner generic's `extract_slice` operands has a size-1 leading
//     axis dropped (rank-reducing slice, source `tensor<8x16x32xf32>`).

// CHECK: func.func @multi_r_noncontig__v0(
// CHECK-SAME: afir.reduce_template = "Common"
// The rank-reducing extract_slice (3-D → 2-D) is the load-bearing structural
// witness that the peel happened.  Its `static_sizes` attribute pins the peel
// axis to a literal 1 (printed as `[1, ...]` in the operand list) and its
// result type drops the leading dim.
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] [1, %{{.*}}, %{{.*}}] {{.*}} : tensor<8x16x32xf32> to tensor<?x?xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]

func.func @multi_r_noncontig(%x: tensor<8x16x32xf32>,
                              %init: tensor<16xf32>) -> tensor<16xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(r1, a, r2) -> (r1, a, r2)>,
                     affine_map<(r1, a, r2) -> (a)>],
    iterator_types = ["reduction", "parallel", "reduction"]}
    ins(%x : tensor<8x16x32xf32>)
    outs(%init : tensor<16xf32>) {
  ^bb0(%v: f32, %acc: f32):
    %t = arith.addf %v, %acc : f32
    linalg.yield %t : f32
  } -> tensor<16xf32>
  return %r : tensor<16xf32>
}
