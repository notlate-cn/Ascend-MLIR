// RUN: afir-opt %s --ascendc-prepare-for-emit 2>&1 | FileCheck %s
//
// Phase B: PrepareForEmit reads tiling.infos and uses actual param names
// (XBLOCK / XBLOCK_SUB) instead of the Phase A positional defaults (TB_M / TB_N).

// CHECK:      func.func @test_func(
// CHECK-SAME:   %{{[^ ,)]*}}: memref<?xf32>
// CHECK-SAME:   %{{[^ ,)]*}}: memref<?x!emitasc.py_struct<"TilingData"
// CHECK:      emitasc.member %{{.*}} "XBLOCK"
// CHECK:      emitasc.member %{{.*}} "XBLOCK_SUB"
// CHECK-NOT:  TB_M
// CHECK-NOT:  TB_N

module attributes {
  vector_plan.tiling_infos = [{
    fields = [
      {abi_index = 0 : i32, arg_index = 1 : i32, default_value = 128 : i64,
       kind = "tunable", name = "XBLOCK"},
      {abi_index = 1 : i32, arg_index = 2 : i32, default_value = 16 : i64,
       kind = "tunable", name = "XBLOCK_SUB"}
    ],
    kernel_id = "test_func"
  }]
} {
  // Post-bufferize form: tensor args are now memref, tiling args are index.
  // The body is empty (no memref.dim uses) so TilingData has only tile fields.
  func.func @test_func(%buf: memref<?xf32>,
                        %xblock: index,
                        %xblock_sub: index) {
    return
  }
}
