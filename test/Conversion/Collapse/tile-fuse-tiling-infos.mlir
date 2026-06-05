// RUN: afir-opt %s --auto-fuse-tile-fuse 2>&1 | FileCheck %s

// CHECK: auto_fuse.tiling_infos
// CHECK-SAME: block_dim_expr = "ceil(1024/XBLOCK)"
// P6a: TileConstraint emission — XBLOCK_SUB | XBLOCK divides; conservative
// LeBytes footprint check against the SoC UB capacity.  (The tail-offset 32B
// reject constraint was dropped: the ragged tail's GM store now goes through
// DataCopyPad, which handles unaligned f16 tail offset/length.)
// CHECK-SAME: constraints = [{kind = "divides", lhs = "XBLOCK_SUB", rhs = "XBLOCK"}, {kind = "le_bytes", lhs = "((12) * XBLOCK_SUB)", rhs = "196608"}]
// Schema v2: fields lose `abi_index` (array order is the ABI order);
// tunables keep arg_index / axis_size / default_value / kind / name.
// CHECK-SAME: fields = [
// CHECK-SAME: arg_index = 2 : i32, axis_size = 1024 : i64, default_value = 128 : i64, kind = "tunable", name = "XBLOCK"
// CHECK-SAME: arg_index = 3 : i32, axis_size = 1024 : i64, default_value = 16 : i64, kind = "tunable", name = "XBLOCK_SUB"
// CHECK-SAME: kernel_id = "pointwise__v0"
// CHECK-SAME: schema_version = 2

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
