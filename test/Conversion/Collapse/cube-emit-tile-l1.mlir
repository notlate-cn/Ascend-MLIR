// RUN: afir-opt %s --vector-plan-tile-fuse --canonicalize 2>&1 | FileCheck %s
//
// CV-fusion Phase 4a+4b of [[af-cv-fusion-port]]: level-1 + level-2 cube
// tile, ascendc annotations, post-canonicalize DCE.
//
// CubeEmitter calls `scf::tileConsumerAndFuseProducersUsingSCF` twice on the
// relu epilogue: first with [XBLOCK_M, XBLOCK_N] (outer multicore tile),
// then with [M_INNER, N_INNER] on the level-1 result (inner intra-block
// tile, sized for the cube fragment 16×16).  K stays untiled until Phase 4c.
//
// Outer 2 loops get `ascendc.parallel`; inner 2 loops are bare.  The
// innermost tiled matmul gets `ascendc.unit = "AiCore.Cube"` and the inner
// tiled relu gets `"AiCore.Vector"`.  Original untiled + level-1-stale
// linalg ops are dead after replaceAllUsesWith; downstream canonicalize
// (this RUN line) DCEs them so only the level-2 tiled ops survive.

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

// 4-level scf.for nest emitted (M_outer × N_outer × M_inner × N_inner).
// Innermost body: tiled matmul + tiled relu with correct ascendc.unit.
// Outermost 2 loops annotated `ascendc.parallel`; inner 2 are bare.
// Outermost loop also carries Phase-4c dataflow strings
// (`ascendc.prologue` GM→A1/B1, `ascendc.epilogue` VECOUT→GM) consumed by
// AscendCBufferPlacement.
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.matmul {ascendc.unit = "AiCore.Cube"}
// CHECK: linalg.generic {{.*}}ascendc.unit = "AiCore.Vector"
// CHECK: } {ascendc.parallel}
// CHECK: ascendc.epilogue = "result:VECOUT->GM"
// CHECK-SAME: ascendc.parallel
// CHECK-SAME: ascendc.prologue = "lhs:GM->A1,rhs:GM->B1"
