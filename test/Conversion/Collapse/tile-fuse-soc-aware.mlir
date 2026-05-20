// RUN: afir-opt %s --auto-fuse-tile-fuse 2>&1 | FileCheck %s --check-prefix=B1
// RUN: afir-opt %s --auto-fuse-tile-fuse=soc=Ascend310B 2>&1 | FileCheck %s --check-prefix=B310B
//
// P6d: --soc threads into TilePlanGen and the LeBytes constraint's UB
// capacity rhs comes from SocSpec, not a hardcoded 192 KiB.

// B1:     constraints = [{kind = "divides", lhs = "XBLOCK_SUB", rhs = "XBLOCK"}, {kind = "divides", lhs = "32", rhs = "((1024 - XBLOCK_SUB) * 4)"}, {kind = "le_bytes", lhs = "((12) * XBLOCK_SUB)", rhs = "196608"}]
// B310B:  constraints = [{kind = "divides", lhs = "XBLOCK_SUB", rhs = "XBLOCK"}, {kind = "divides", lhs = "32", rhs = "((1024 - XBLOCK_SUB) * 4)"}, {kind = "le_bytes", lhs = "((12) * XBLOCK_SUB)", rhs = "120832"}]

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
