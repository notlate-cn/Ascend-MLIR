// RUN: afir-opt %s --auto-fuse-codegen 2>&1 | FileCheck %s
//
// Regression: elementwise-into-reduce (out[d0,d1] = sum_d2(a+b)).  The fused
// reduce body lowers to two vector adds on VECCALC TBufs:
//   add_l2 %t,   a, b      // t = a + b          (fresh TBuf)
//   add_l2 %acc, %acc, %t  // acc += t           (accumulator)
// The accumulate reads %t (and %acc), both written by preceding vector ops into
// TBufs that have no queue (EnQue/DeQue) sync.  Without an explicit barrier the
// read races the write on real hardware across multi-repeat tiles (the
// simulator hides it) — combo-elewise-reduce-e2e produced ~0 on the 910C.
// ComputeConversion must emit a PipeBarrier between the producing add and the
// consuming accumulate.  See lib/Conversion/LinalgToAscendC/ComputeConversion.cpp.

// CHECK-LABEL: func.func @combo_raw__v0(
// The first add writes t into a fresh TBuf; the accumulate then reads it, but
// only AFTER a pipe_barrier orders the write before the read.
// CHECK:      ascendc.add_l2 %[[T:[0-9]+]], %{{[0-9]+}}, %{{[0-9]+}},
// CHECK-NEXT: ascendc.pipe_barrier pipe_all
// CHECK-NEXT: ascendc.add_l2 %{{[0-9]+}}, %{{[0-9]+}}, %[[T]],
// CHECK:      ascendc.reduce_sum_2d_l2

#map_full   = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @combo_raw(%a: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>,
                     %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %a, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %a, %c2 : tensor<?x?x?xf32>
  %ey = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %y = linalg.generic {
      indexing_maps = [#map_full, #map_full, #map_full],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%ey : tensor<?x?x?xf32>) {
  ^bb0(%av: f32, %bv: f32, %_: f32):
    %s = arith.addf %av, %bv : f32
    linalg.yield %s : f32
  } -> tensor<?x?x?xf32>
  %out = linalg.generic {
      indexing_maps = [#map_full, #map_reduce],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%y : tensor<?x?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %v = arith.addf %acc, %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?xf32>
  return %out : tensor<?x?xf32>
}
