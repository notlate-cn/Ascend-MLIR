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

// -----

// v2 schema path: PackTilingData should consume vector_plan.tiling_infos
// directly and emit field names verbatim — no argToOriginal redirect.
// CHECK-LABEL: func.func @v2_transpose
// CHECK: emitasc.py_struct<"TilingData", [i64, i64], ["XBLOCK", "dim_arg2_1"]>
module attributes {vector_plan.tiling_infos = [{
    kernel_id = "v2_transpose",
    schema_version = 2 : i32,
    fields = [
      {name = "XBLOCK", kind = "tunable", axis_size = 32 : i64,
       default_value = 16 : i64, arg_index = 1 : i32},
      {name = "dim_arg2_1", kind = "shape_derived",
       source_arg = 2 : i32, source_dim = 1 : i32}
    ],
    args = [
      {mlir_index = 0 : i32, role = "input",  network_index = 0 : i32},
      {mlir_index = 1 : i32, role = "tile_param", name = "XBLOCK"},
      {mlir_index = 2 : i32, role = "output", result_index = 0 : i32,
       shape_expr = ["arg0_dim1", "arg0_dim0"]}
    ]
  }]} {
  func.func @v2_transpose(%arg0: memref<16x32xf16>, %arg1: index,
                          %arg2: memref<32x16xf16, strided<[?, 1], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %d = memref.dim %arg2, %c1 : memref<32x16xf16, strided<[?, 1], offset: ?>>
    return
  }
}
