// RUN: afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s
//
// VectorGroup end-to-end codegen: 1D pointwise through the full
// --vector-plan-codegen pipeline. Verifies AiCore dispatch (get_block_idx),
// Phase B names in TilingData (XBLOCK/XBLOCK_SUB), and kernel attributes.

// CHECK:      func.func @pointwise(
// CHECK-SAME: ascendc.aicore
// CHECK:      emitasc.member %{{.*}} "XBLOCK"
// CHECK:      emitasc.member %{{.*}} "XBLOCK_SUB"
// CHECK:      ascendc.get_block_idx
// CHECK-NOT:  "TB_M"
// CHECK-NOT:  "TB_N"

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
