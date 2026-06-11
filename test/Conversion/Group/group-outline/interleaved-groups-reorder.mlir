// RUN: afir-opt --auto-fuse-group-outline %s | FileCheck %s

// Group analysis may leave a group's members interleaved with another group's
// in block order, even when the group graph is a DAG (so each group is convex).
// The outliner reorders the coordinator to cluster each group's members
// contiguously, then outlines successfully.  Here group 0 owns ops at topo 0
// and 2 with group 1's op interleaved at topo 1; the three ops are mutually
// independent, so reordering to {0,0,1} is valid and both kernels are emitted.

#map = affine_map<(d0) -> (d0)>

func.func @interleaved(%a: tensor<8xf16>, %b: tensor<8xf16>, %c: tensor<8xf16>,
                       %i0: tensor<8xf16>, %i1: tensor<8xf16>, %i2: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>, tensor<8xf16>) {
  %g0a = linalg.generic {
    indexing_maps = [#map, #map], iterator_types = ["parallel"],
    auto_fuse.group_id = 0 : i32, auto_fuse.topo_index = 0 : i32
  } ins(%a : tensor<8xf16>) outs(%i0 : tensor<8xf16>) {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %g1 = linalg.generic {
    indexing_maps = [#map, #map], iterator_types = ["parallel"],
    auto_fuse.group_id = 1 : i32, auto_fuse.topo_index = 1 : i32
  } ins(%b : tensor<8xf16>) outs(%i1 : tensor<8xf16>) {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %g0b = linalg.generic {
    indexing_maps = [#map, #map], iterator_types = ["parallel"],
    auto_fuse.group_id = 0 : i32, auto_fuse.topo_index = 2 : i32
  } ins(%c : tensor<8xf16>) outs(%i2 : tensor<8xf16>) {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  return %g0a, %g1, %g0b : tensor<8xf16>, tensor<8xf16>, tensor<8xf16>
}

// Both groups outline (no "not contiguous" error).
// CHECK-DAG: func.func private @kernel_group0
// CHECK-DAG: func.func private @kernel_group1
// CHECK: func.func @interleaved
