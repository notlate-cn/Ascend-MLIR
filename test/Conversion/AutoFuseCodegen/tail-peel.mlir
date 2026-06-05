// RUN: afir-opt %s --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:   --auto-fuse-tile-fuse | FileCheck %s
//
// Verify the tail-peel IR shape:
//   - inner scf.for ub is `mainInnerUb = (remaining / step) * step`
//   - scf.if (mainInnerUb < remaining) follows it
//   - scf.if then-block uses HONEST size `remaining - mainInnerUb` at offset
//     `outer_iv + mainInnerUb` (ragged-tail, NOT overlap `extent - T`)
//   - scf.if else-block yields innermost-for results unchanged

func.func @reduce_tail(%x: tensor<?x?x?xf32>,
                        %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %out = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                       affine_map<(d0,d1,d2)->(d0,d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%x : tensor<?x?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %v = arith.addf %acc, %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?xf32>
  return %out : tensor<?x?xf32>
}

// XBLOCK_SUB-tunable arg is %arg3 (after the two function inputs).
// extent = D0*D1 (rows), parentStep = XBLOCK (%arg2).
// Peel structure inside the outer scf.for body:
//
//   remaining     = arith.minsi %XBLOCK %(extent - outer_iv)
//   mainInnerUb   = (remaining / XBLOCK_SUB) * XBLOCK_SUB
//   inner scf.for ub = mainInnerUb
//   scf.if (mainInnerUb < remaining):
//     // overlap-tail: composed IV = extent - XBLOCK_SUB (static slice size)
//     ...
//
// CHECK-LABEL: func.func @reduce_tail
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %arg2
//
// remaining = min(XBLOCK, extent - outer_iv)
// CHECK: %[[REMAINING:.+]] = arith.minsi %arg2
//
// mainInnerUb = (remaining / XBLOCK_SUB) * XBLOCK_SUB
// CHECK: %[[Q:.+]] = arith.divsi %[[REMAINING]], %arg3
// CHECK: %[[MAIN_UB:.+]] = arith.muli %[[Q]], %arg3
//
// Inner scf.for with ub = mainInnerUb (not XBLOCK).
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[MAIN_UB]] step %arg3
// CHECK:   tensor.extract_slice
// CHECK:   linalg.generic
// CHECK:   tensor.insert_slice
// CHECK:   scf.yield
//
// Tail body: ragged-tail uses the HONEST offset `outer_iv + mainInnerUb` and
// the HONEST size `remaining - mainInnerUb` (NOT `extent - T` / static `%arg3`).
// No overlap, no recompute.
//
// CHECK: %[[COND:.+]] = arith.cmpi slt, %[[MAIN_UB]], %[[REMAINING]]
// CHECK: %[[TAILRES:.+]] = scf.if %[[COND]]
//
// honest tail size = remaining - mainInnerUb
// CHECK: %[[TAILSZ:.+]] = arith.subi %[[REMAINING]], %[[MAIN_UB]]
// honest tail offset = outer_iv + mainInnerUb
// CHECK: %[[TAILOFF:.+]] = arith.addi %{{.*}}, %[[MAIN_UB]]
//
// extract_slice carries honest offset + honest (dynamic) size.
// CHECK: tensor.extract_slice %{{.*}}[%[[TAILOFF]]] [%[[TAILSZ]]]
// CHECK: linalg.generic
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%[[TAILOFF]]] [%[[TAILSZ]]]
// CHECK: scf.yield
//
// else branch yields the inner-for results unchanged.
// CHECK: } else {
// CHECK: scf.yield
// CHECK: }
