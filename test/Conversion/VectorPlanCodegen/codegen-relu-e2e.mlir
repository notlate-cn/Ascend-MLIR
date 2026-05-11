// RUN: afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s --check-prefix=MLIR
// RUN: afir-opt %s --vector-plan-codegen 2>&1 | afir-translate -mlir-to-cann | FileCheck %s --check-prefix=CANN
//
// ReLU (max(x, 0)) end-to-end through --vector-plan-codegen + afir-translate.
// Verifies the full stack for a parallel generic with an arith.maximumf body:
//   GM input  →  VECIN (DataCopy)
//   zero tile →  VECCALC (Duplicate)
//   max        →  VECCALC (Max)
//   VECOUT     →  GM (DataCopy)

// MLIR: func.func @relu(
// MLIR-SAME: ascendc.aicore
// MLIR: emitasc.member {{.*}} "XBLOCK"
// MLIR: emitasc.member {{.*}} "XBLOCK_SUB"
// MLIR: ascendc.get_block_idx
// MLIR: ascendc.data_copy_l2 {{.*}} : !ascendc.local_tensor
// MLIR: ascendc.duplicate_l2
// MLIR: ascendc.max_l2
// MLIR: ascendc.data_copy_l2 {{.*}} : !ascendc.global_tensor
// MLIR-NOT: linalg.generic

// CANN: struct TilingData
// CANN: XBLOCK
// CANN: __aicore__ void relu(
// CANN: GetBlockIdx()
// CANN: SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v1) + {{.*}})
// CANN: DataCopy({{.*}}, {{.*}}, {{.*}})
// CANN: Duplicate({{.*}}, c0_f32,
// CANN: Max(
// Output write (GM destination arg index is ABI-dependent — don't pin it).
// CANN: SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v{{[0-9]+}}) + {{.*}})

func.func @relu(%a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%a : tensor<1024xf32>) outs(%b : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %zero = arith.constant 0.000000e+00 : f32
    %r = arith.maximumf %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
