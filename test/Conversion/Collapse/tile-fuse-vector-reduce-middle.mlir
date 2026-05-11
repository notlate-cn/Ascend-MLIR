// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// 3-D reduction on the MIDDLE axis: d0(parallel), d1(reduction), d2(parallel).
// d0 and d2 cannot collapse (reduction axis between them), so >=2 parallel
// axes are separated by a reduction axis → the "split-parallel" policy:
//   - only XBLOCK is a tunable (no XBLOCK_SUB);
//   - d0 is block-dispatched and walked one row at a time (inner step = 1);
//   - d1 (reduction) and d2 stay full → operand slice x[row, :, :] and output
//     slice out[row, :] are both contiguous in row-major layout.

// CHECK: func.func @reduce_mid(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-NOT: vector_plan.default_tile_size = 16

// Outer block-dispatch loop.
// CHECK: scf.for %[[OUTER:[^ ]*]] = %{{.*}} to %{{.*}} step %[[XBLOCK]]

// Inner row loop, step 1 (one d0 row per tile body).
// CHECK: %[[REM:[^ ]*]] = arith.minsi %[[XBLOCK]]
// CHECK: scf.for %[[ROW:[^ ]*]] = %{{.*}} to %{{.*}} step %[[C1:[^ ]*]] iter_args
// CHECK: %[[IDX:[^ ]*]] = arith.addi %[[OUTER]], %[[ROW]]

// Operand x[idx, :, :] : [1, D1, D2] — contiguous.
// CHECK: tensor.extract_slice %{{.*}}[%[[IDX]], %{{.*}}, %{{.*}}] [%[[C1]], %{{.*}}, %{{.*}}]
// Output out[idx, :] : [1, D2] — contiguous.
// CHECK: tensor.extract_slice %{{.*}}[%[[IDX]], %{{.*}}] [%[[C1]], %{{.*}}]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK: tensor.insert_slice
// CHECK: scf.yield
// CHECK: } {ascendc.parallel}

func.func @reduce_mid(%x: tensor<?x?x?xf32>,
                      %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d2)>],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%x : tensor<?x?x?xf32>)
    outs(%init : tensor<?x?xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %v = arith.addf %acc, %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?xf32>
  return %out : tensor<?x?xf32>
}
