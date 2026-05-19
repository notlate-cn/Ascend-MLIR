// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// CV-fusion Phase 4a of [[af-cv-fusion-port]]: level-1 cube tile + annotate.
//
// CubeEmitter calls `scf::tileConsumerAndFuseProducersUsingSCF` on the relu
// epilogue with tile sizes [XBLOCK_M, XBLOCK_N] (K untiled), fusing the
// matmul producer into the inner of two scf.for loops.  The outer 2 loops
// get `ascendc.parallel` (multicore dispatch); the tiled matmul gets
// `ascendc.unit = "AiCore.Cube"` and the tiled relu gets `"AiCore.Vector"`.
//
// Level-2 (inner M/N) and level-3 (K) tile + the `ascendc.prologue/epilogue`
// dataflow string annotations land in Phase 4b/c.

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

// Outer 2-level scf.for nest emitted; both annotated ascendc.parallel.
// Lit checks follow input order: scf.for opens → body (matmul + vec) → closes.
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.matmul {ascendc.unit = "AiCore.Cube"}
// CHECK: linalg.generic {{.*}}ascendc.unit = "AiCore.Vector"
// CHECK: } {ascendc.parallel}
// CHECK: } {ascendc.parallel}
