// RUN: afir-opt --auto-fuse-group-analysis %s | FileCheck %s
//
// Phase 1 of [[af-cv-fusion-port]] — enable CV (Cube + Vector) fusion at the
// GroupAnalysis layer.  GroupAnalysisPass.cpp:147-151 used to gate
// `oneCube` with `canFuse = false`; now it dispatches to
// `canFuseCubeEpilogue` (already implemented in CanFuse.cpp:286).
//
// linalg.matmul (Cube) → linalg.generic relu (Vector, all-parallel iter,
// matches matmul output rank, single consumer of matmul result) → both
// ops land in the SAME group.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @mm_relu(%a: tensor<32x16xf16>,
                    %b: tensor<16x64xf16>,
                    %init: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %c = linalg.matmul
    ins(%a, %b : tensor<32x16xf16>, tensor<16x64xf16>)
    outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>

  %empty = tensor.empty() : tensor<32x64xf32>
  %r = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%c : tensor<32x64xf32>) outs(%empty : tensor<32x64xf32>) {
  ^bb0(%v: f32, %_: f32):
    %z = arith.constant 0.0 : f32
    %t = arith.maximumf %v, %z : f32
    linalg.yield %t : f32
  } -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}

// Both matmul and the trailing relu generic must share the same group_id.
// CHECK: linalg.matmul
// CHECK-SAME: auto_fuse.group_id = [[G:[0-9]+]]
// CHECK: linalg.generic
// CHECK-SAME: auto_fuse.group_id = [[G]]
