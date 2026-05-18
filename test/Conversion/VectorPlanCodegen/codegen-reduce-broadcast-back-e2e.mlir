// RUN: afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s
//
// "layernorm-ish" pattern: a reduce whose result s[a] is broadcast back over
// the reduced axis and fed to an elementwise op, all in one kernel group.
// The reduce result is an intra-group intermediate that GroupEmitter would
// otherwise allocate as a `bufferization.alloc_tensor {memory_space = 11}`
// (→ `memref.alloc : memref<?xf32, 11>`).  The reduce path must register that
// reduce result as the live tensor for the output memref so the downstream
// elementwise broadcast reads it directly (broadcast_l2 of the reduce result)
// instead of treating the VECCALC alloc as a GM source — the latter leaves a
// `memref.alloc : memref<?xf32, 11>` alive that has no printer (translate fails).

// CHECK:      func.func @c1_ln__v0(
// CHECK-SAME: cann.num_inputs = 1
// No space-11 memref.alloc must survive into the AscendC IR.
// CHECK-NOT:  memref.alloc{{.*}}memref<?xf32, 11>
// The reduce result feeds the broadcast which feeds the subtract.
// CHECK:      ascendc.reduce_sum_2d_l2 %[[RED:.*]], %{{.*}} {layout
// CHECK:      ascendc.broadcast_l2 %{{.*}}, %[[RED]],
// CHECK:      ascendc.sub_l2

func.func @c1_ln(%x: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %c0 = arith.constant 0.000000e+00 : f32
  %s_empty = tensor.empty() : tensor<16xf32>
  %s_init = linalg.fill ins(%c0 : f32) outs(%s_empty : tensor<16xf32>) -> tensor<16xf32>
  %s = linalg.generic {
    indexing_maps = [affine_map<(a, b) -> (a, b)>, affine_map<(a, b) -> (a)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%x : tensor<16x64xf32>) outs(%s_init : tensor<16xf32>) {
  ^bb0(%i: f32, %acc: f32):
    %t = arith.addf %i, %acc : f32
    linalg.yield %t : f32
  } -> tensor<16xf32>
  %o = tensor.empty() : tensor<16x64xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(a, b) -> (a, b)>, affine_map<(a, b) -> (a)>, affine_map<(a, b) -> (a, b)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x, %s : tensor<16x64xf32>, tensor<16xf32>) outs(%o : tensor<16x64xf32>) {
  ^bb0(%xi: f32, %si: f32, %oi: f32):
    %t = arith.subf %xi, %si : f32
    linalg.yield %t : f32
  } -> tensor<16x64xf32>
  return %r : tensor<16x64xf32>
}
