// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// 2-D reduction: d0(parallel), d1(reduction, no split).
// No collapse (different roles). Reduction axis d1 has no loop IV → full-dim slice.
// Expected: XBLOCK + XBLOCK_SUB args. Outer + inner scf.for. extract_slice of %a takes full d1.

// CHECK: func.func @reduce(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64}

// CHECK: scf.for %[[OUTER:[^ ]*]] = %{{.*}} to %{{.*}} step %[[XBLOCK]]
//
// Tail-peel: inner ub = (remaining / XBLOCK_SUB) * XBLOCK_SUB.
// CHECK: %[[REM:[^ ]*]] = arith.minsi %[[XBLOCK]]
// CHECK: %[[Q:[^ ]*]] = arith.divsi %[[REM]], %[[XBLOCK_SUB]]
// CHECK: %[[MAINUB:[^ ]*]] = arith.muli %[[Q]], %[[XBLOCK_SUB]]
// CHECK: scf.for %[[INNER:[^ ]*]] = %{{.*}} to %[[MAINUB]] step %[[XBLOCK_SUB]]
// The reduction axis (d1) has no loop IV → full-dim slice in the extract.
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
// CHECK: tensor.insert_slice
// CHECK: scf.yield
//
// Overlap-tail in scf.if (mainUb < remaining), then else branch.
// CHECK: arith.cmpi slt, %[[MAINUB]], %[[REM]]
// CHECK: scf.if
// CHECK: linalg.generic
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: } {ascendc.parallel}

func.func @reduce(%a: tensor<1024x512xf32>,
                  %c: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%a : tensor<1024x512xf32>)
    outs(%c : tensor<1024xf32>) {
  ^bb0(%a0: f32, %acc: f32):
    %add = arith.addf %a0, %acc : f32
    linalg.yield %add : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
