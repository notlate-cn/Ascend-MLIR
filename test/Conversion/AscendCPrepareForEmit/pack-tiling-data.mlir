// RUN: afir-opt %s --split-input-file --verify-diagnostics --ascendc-pack-tiling-data | FileCheck %s
//
// Verify:
//  - Tiling args replaced by emitasc.member reads
//  - memref.dim replaced by emitasc.member read
//  - TilingData GM pointer arg added
//  - No i64/index tiling block args remain

// CHECK: func.func @relu(
// CHECK-SAME: memref<1024xf32>
// CHECK-SAME: memref<1024xf32>
// CHECK-SAME: !emitasc.py_struct<"TilingData"
// CHECK: emitasc.member {{.*}} "XBLOCK"
// CHECK: emitasc.member {{.*}} "XBLOCK_SUB"
// CHECK: emitasc.member {{.*}} "dim_arg0_0"
// CHECK: emitasc.member {{.*}} "dim_arg1_0"
// CHECK-NOT: TB_M

module attributes {
  vector_plan.tiling_infos = [{
    fields = [
      {abi_index = 0 : i32, arg_index = 2 : i32, default_value = 128 : i64,
       kind = "tunable", name = "XBLOCK"},
      {abi_index = 1 : i32, arg_index = 3 : i32, default_value = 16 : i64,
       kind = "tunable", name = "XBLOCK_SUB"}
    ],
    kernel_id = "relu"
  }]
} {
  func.func @relu(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>,
                  %xblock: index, %xblock_sub: index) {
    %c0 = arith.constant 0 : index
    %d0 = memref.dim %arg0, %c0 : memref<1024xf32>
    %d1 = memref.dim %arg1, %c0 : memref<1024xf32>
    %val = memref.load %arg0[%d0] : memref<1024xf32>
    return
  }
}

// -----

// Verify that the pass emits an error when vector_plan.tiling_infos is absent.

module {
  // expected-error@+1 {{PackTilingData: vector_plan.tiling_infos not found}}
  func.func @no_tiling(%arg0: index) {
    return
  }
}
