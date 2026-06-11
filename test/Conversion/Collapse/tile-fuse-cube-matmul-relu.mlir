// RUN: afir-opt %s --auto-fuse-tile-fuse 2>&1 | FileCheck %s
//
// CV-fusion Phase 2: cube TilePlan structure ([[af-cv-fusion-port]]).
//
// `linalg.matmul + linalg.generic(relu)` after Phase 1's canFuseCubeEpilogue
// lands in a single Cube group.  Phase 2 TileFusePass detects this in
// `collapseGroup` (CollapsedGroupInfo.kind == Cube), enumerates exactly one
// cube draft (`cubeKind = MatmulVecFuse`), and `buildCubePlan` emits:
//   - 5 cube tunable i64 func args:
//     XBLOCK_M, M_INNER, XBLOCK_N, N_INNER, K_INNER
//   - `ascendc.kernel_kind = "mix"` (gates downstream mix-pipeline passes)
//   - `afir.cube_kind = "MatmulVecFuse"` (records the picked template)
//   - schema v2 `auto_fuse.tiling_infos` with the 5 tunable fields
// The matmul + generic linalg ops are LEFT IN PLACE for Phase-4 emission.

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

// Module-level auto_fuse.tiling_infos schema v2 with the 5 cube tunables.
// Emitted on the module attr line (before the func) — CHECK directives match
// in order, so schema_version checks come first.
// CHECK-DAG: name = "XBLOCK_M"
// CHECK-DAG: name = "M_INNER"
// CHECK-DAG: name = "XBLOCK_N"
// CHECK-DAG: name = "N_INNER"
// CHECK-DAG: name = "K_INNER"
// CHECK-DAG: schema_version = 2

// Mix-kernel attrs on the renamed func.
// CHECK-LABEL: func.func @mm_relu__v0
// CHECK-SAME: afir.cube_kind = "MatmulVecFuse"
// CHECK-SAME: ascendc.kernel_kind = "mix"

// Linalg ops survive untouched — Phase 4 will emit the cube nest from them.
// CHECK: linalg.matmul
// CHECK: linalg.generic
