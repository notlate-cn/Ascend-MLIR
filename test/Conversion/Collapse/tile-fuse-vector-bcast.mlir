// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// BCast: d0(1024)+d1(512) collapse → d0'(524288); d2(16) is BCast (b misses d2).
// Loop order: XBLOCK(outer) → BCAST_0(step=1, ub=16) → XBLOCK_SUB(inner).
// %b_collapsed does not depend on the BCast axis → its extract_slice is hoisted
// before the BCAST_0 loop.

// CHECK: func.func @bcast_op(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64}

// Outer XBLOCK loop (ascendc.parallel attribute appears on closing brace line)
// CHECK: scf.for %[[OUTER:[^ ]*]] = %{{.*}} to %{{.*}} step %[[XBLOCK]]

// %b slice hoisted here (before BCAST loop, using outer IV and XBLOCK size)
// CHECK: tensor.extract_slice %{{.*}}[%[[OUTER]]] [%[[XBLOCK]]]

// BCast loop (no {ascendc.parallel})
// CHECK: scf.for %[[BCAST:[^ ]*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NOT: {ascendc.parallel}

// Tail-peel: inner ub = (remaining / XBLOCK_SUB) * XBLOCK_SUB.
// CHECK: %[[REM:[^ ]*]] = arith.minsi %[[XBLOCK]]
// CHECK: %[[Q:[^ ]*]] = arith.divsi %[[REM]], %[[XBLOCK_SUB]]
// CHECK: %[[MAINUB:[^ ]*]] = arith.muli %[[Q]], %[[XBLOCK_SUB]]

// Inner XBLOCK_SUB loop (ub = mainUb, not XBLOCK)
// CHECK: scf.for %[[INNER:[^ ]*]] = %{{.*}} to %[[MAINUB]] step %[[XBLOCK_SUB]]
// CHECK: linalg.generic
// CHECK: scf.yield

// Overlap-tail scf.if (mainUb < remaining) inside the BCast loop.
// CHECK: arith.cmpi slt, %[[MAINUB]], %[[REM]]
// CHECK: scf.if
// CHECK: linalg.generic
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield

func.func @bcast_op(%a: tensor<1024x512x16xf32>,
                    %b: tensor<1024x512xf32>,
                    %c: tensor<1024x512x16xf32>) -> tensor<1024x512x16xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%a, %b : tensor<1024x512x16xf32>, tensor<1024x512xf32>)
    outs(%c : tensor<1024x512x16xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %mul = arith.mulf %a0, %b0 : f32
    linalg.yield %mul : f32
  } -> tensor<1024x512x16xf32>
  return %result : tensor<1024x512x16xf32>
}
