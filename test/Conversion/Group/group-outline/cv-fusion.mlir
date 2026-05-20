// RUN: afir-opt %s --auto-fuse-group-analysis --auto-fuse-group-outline | FileCheck %s
//
// CV-fusion Phase 3 of [[af-cv-fusion-port]].  Phase 1's canFuseCubeEpilogue
// merges `linalg.matmul + linalg.generic relu` into one Cube group; existing
// GroupOutline naturally outlines them into one kernel func (no Phase-3 code
// change for the body itself).  Phase 3 adds:
//   - `auto_fuse.kind = "Cube"` attr on the outlined kernel func (Cube vs
//     Vector dispatch hint for downstream passes).
//   - Lit pin so the cube outline shape is locked.

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

// One outlined kernel func — both linalg ops inside, stamped Cube kind.
// CHECK-LABEL: func.func private @kernel_group0
// CHECK-SAME: attributes {auto_fuse.kind = "Cube"}
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: return

// Coordinator func calls the single kernel once.
// CHECK-LABEL: func.func @mm_relu
// CHECK: call @kernel_group0
// CHECK: return
