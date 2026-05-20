// RUN: afir-opt %s --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:   --afir-symbolize-shapes --auto-fuse-tile-fuse | FileCheck %s

// With afir-symbolize-shapes in front of tile-fuse, the post-collapse symbolic
// axis extents are recorded as afir.axis_extents on the kernel func and the
// symbol registry afir.dim_symbols survives.  Here the two leading parallel
// dims d0,d1 collapse into one axis whose extent is the product (s0*s1); the
// trailing reduction dim d2 stays s2.

// CHECK-LABEL: func.func @collapsed_reduce
// CHECK-SAME:  afir.axis_extents = ["(s0*s1)", "s2"]
// CHECK-SAME:  afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}, {arg = 0 : i64, dim = 2 : i64, id = 2 : i64}]
func.func @collapsed_reduce(%a: tensor<?x?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, affine_map<(d0,d1,d2)->(d0,d1)>],
      iterator_types = ["parallel","parallel","reduction"]}
      ins(%a : tensor<?x?x?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %acc, %x : f32
      linalg.yield %s : f32
  } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
