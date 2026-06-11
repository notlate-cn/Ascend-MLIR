// RUN: not afir-opt --auto-fuse-group-outline %s 2>&1 | FileCheck %s

// Safety net: a genuinely cyclic grouping (group 0's members straddle group 1,
// which both consumes from and feeds back into group 0: m1 -> e -> m2 with
// m1,m2 in group 0 and e in group 1) is NOT a DAG, so it cannot be reordered to
// contiguity.  The outliner must fail cleanly with a diagnostic rather than
// segfaulting on dangling handles.  (Group analysis should never produce this
// after the glue-aware cycle check; this guards against regressions and
// hand-written IR.)

#map = affine_map<(d0) -> (d0)>

func.func @cyclic(%a: tensor<8xf16>,
                  %i0: tensor<8xf16>, %i1: tensor<8xf16>, %i2: tensor<8xf16>)
    -> tensor<8xf16> {
  %m1 = linalg.generic {
    indexing_maps = [#map, #map], iterator_types = ["parallel"],
    auto_fuse.group_id = 0 : i32, auto_fuse.topo_index = 0 : i32
  } ins(%a : tensor<8xf16>) outs(%i0 : tensor<8xf16>) {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %e = linalg.generic {
    indexing_maps = [#map, #map], iterator_types = ["parallel"],
    auto_fuse.group_id = 1 : i32, auto_fuse.topo_index = 1 : i32
  } ins(%m1 : tensor<8xf16>) outs(%i1 : tensor<8xf16>) {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  %m2 = linalg.generic {
    indexing_maps = [#map, #map], iterator_types = ["parallel"],
    auto_fuse.group_id = 0 : i32, auto_fuse.topo_index = 2 : i32
  } ins(%e : tensor<8xf16>) outs(%i2 : tensor<8xf16>) {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>

  return %m2 : tensor<8xf16>
}

// CHECK: error: {{.*}}not contiguous
