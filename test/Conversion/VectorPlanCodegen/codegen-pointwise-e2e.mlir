// RUN: afir-opt %s \
// RUN:   --vector-plan-tile-fuse \
// RUN:   "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
// RUN:   --annotate-ascendc-kernel-kind \
// RUN:   --cse \
// RUN:   --ascendc-buffer-placement \
// RUN:   --linalg-to-ascendc \
// RUN:   --canonicalize --cse \
// RUN:   --ascendc-prepare-for-emit \
// RUN:   --canonicalize-cann-signature \
// RUN:   2>&1 | FileCheck %s
//
// VectorGroup end-to-end codegen: 1D pointwise through TileFuse -> codegen.
// Verifies: Phase B names in TilingData (XBLOCK/XBLOCK_SUB), not Phase A defaults.
//
// NOTE: --ascendc-parallelize is omitted because TileFuse emits outer scf.for
// with iter_args (memref threading), which ascendc-parallelize does not yet
// support. That pass expects a no-result parallel loop. Follow-up needed.
// NOTE: This test spells out passes individually rather than invoking
// --vector-plan-codegen, because the registered pipeline includes
// --ascendc-parallelize which crashes on iter_args loops (see above).
// TODO: Once parallelize supports iter_args, replace with --vector-plan-codegen.

// CHECK: func.func @pointwise(
// CHECK-SAME: ascendc.aicore
// CHECK: emitasc.member %{{.*}} "XBLOCK"
// CHECK: emitasc.member %{{.*}} "XBLOCK_SUB"
// CHECK-NOT: "TB_M"
// CHECK-NOT: "TB_N"

func.func @pointwise(%a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%a : tensor<1024xf32>) outs(%b : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
