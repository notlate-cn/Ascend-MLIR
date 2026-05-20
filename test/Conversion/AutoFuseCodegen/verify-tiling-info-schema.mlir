// RUN: afir-opt %s --split-input-file --verify-diagnostics --auto-fuse-verify-tiling-info-schema | FileCheck %s
//
// Cross-check the auto_fuse.tiling_infos v2 schema against the bufferized
// kernel signature.  Positive case passes through unchanged; each negative
// case hard-fails with a specific error.

// ---- Positive: a valid v2 schema (modeled on @v2_transpose) -------------

// CHECK-LABEL: func.func @v2_transpose
module attributes {auto_fuse.tiling_infos = [{
    kernel_id = "v2_transpose",
    schema_version = 2 : i32,
    fields = [
      {name = "XBLOCK", kind = "tunable", axis_size = 32 : i64,
       default_value = 16 : i64, arg_index = 1 : i32},
      {name = "dim_arg2_1", kind = "shape_derived",
       source_arg = 2 : i32, source_dim = 1 : i32}
    ],
    args = [
      {mlir_index = 0 : i32, role = "input",  call_arg_index = 0 : i32},
      {mlir_index = 1 : i32, role = "tile_param", name = "XBLOCK"},
      {mlir_index = 2 : i32, role = "output", result_index = 0 : i32,
       shape_expr = ["arg0_dim1", "arg0_dim0"]}
    ]
  }]} {
  func.func @v2_transpose(%arg0: memref<16x32xf16>, %arg1: index,
                          %arg2: memref<32x16xf16, strided<[?, 1], offset: ?>>) {
    return
  }
}

// -----

// ---- Negative (a): tunable field.arg_index points to a non-TileParam arg.
// The tunable field claims arg_index 0, but arg 0 has role "input".

module attributes {auto_fuse.tiling_infos = [{
    kernel_id = "bad_field_arg",
    schema_version = 2 : i32,
    fields = [
      {name = "XBLOCK", kind = "tunable", axis_size = 32 : i64,
       default_value = 16 : i64, arg_index = 0 : i32}
    ],
    args = [
      {mlir_index = 0 : i32, role = "input",  call_arg_index = 0 : i32},
      {mlir_index = 1 : i32, role = "tile_param", name = "XBLOCK"},
      {mlir_index = 2 : i32, role = "output", result_index = 0 : i32}
    ]
  }]} {
  // expected-error@+2 {{tunable field 'XBLOCK' (arg_index 0) has no matching tile_param SchemaArg}}
  // expected-error@+1 {{tile_param arg mlir_index 1 (name 'XBLOCK') has no matching tunable field}}
  func.func @bad_field_arg(%arg0: memref<16x32xf16>, %arg1: index,
                           %arg2: memref<32x16xf16, strided<[?, 1], offset: ?>>) {
    return
  }
}

// -----

// ---- Negative (b): duplicate result_index on two Output args. ----------

module attributes {auto_fuse.tiling_infos = [{
    kernel_id = "dup_result_index",
    schema_version = 2 : i32,
    fields = [],
    args = [
      {mlir_index = 0 : i32, role = "input",  call_arg_index = 0 : i32},
      {mlir_index = 1 : i32, role = "output", result_index = 0 : i32},
      {mlir_index = 2 : i32, role = "output", result_index = 0 : i32}
    ]
  }]} {
  // expected-error@+1 {{Output resultIndex values must be unique 0..K-1, got [0, 0]}}
  func.func @dup_result_index(%arg0: memref<16x32xf16>,
                              %arg1: memref<16x32xf16, strided<[?, 1], offset: ?>>,
                              %arg2: memref<16x32xf16, strided<[?, 1], offset: ?>>) {
    return
  }
}

// -----

// ---- Negative (c): shape_derived source_arg out of range. --------------

module attributes {auto_fuse.tiling_infos = [{
    kernel_id = "bad_source_arg",
    schema_version = 2 : i32,
    fields = [
      {name = "dim_arg9_1", kind = "shape_derived",
       source_arg = 9 : i32, source_dim = 1 : i32}
    ],
    args = [
      {mlir_index = 0 : i32, role = "input",  call_arg_index = 0 : i32},
      {mlir_index = 1 : i32, role = "output", result_index = 0 : i32}
    ]
  }]} {
  // expected-error@+1 {{shape_derived field 'dim_arg9_1' source_arg 9 out of range (func has 2 args)}}
  func.func @bad_source_arg(%arg0: memref<16x32xf16>,
                            %arg1: memref<32x16xf16, strided<[?, 1], offset: ?>>) {
    return
  }
}
