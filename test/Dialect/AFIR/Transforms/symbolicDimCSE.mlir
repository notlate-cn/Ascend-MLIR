// RUN: afir-opt %s -split-input-file --afir-symbolic-dim-cse | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// Three dims all = s0 (root arg0,dim0) collapse to ONE tensor.dim on %arg0.
// CHECK-LABEL: func.func @merge_same_symbol
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK-NOT: tensor.dim
// CHECK: return %[[D]], %[[D]], %[[D]]
func.func @merge_same_symbol(
    %arg0: tensor<?xf32> {afir.symbolic_shape = "s0"},
    %arg1: tensor<?xf32> {afir.symbolic_shape = "s0"})
    -> (index, index, index)
    attributes {afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}]} {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>)
      attrs = {afir.symbolic_shapes = ["s0"]} {
  ^bb0(%x: f32, %y: f32):
    linalg.yield %x : f32
  } -> tensor<?xf32>
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %d1 = tensor.dim %arg1, %c0 : tensor<?xf32>
  %d2 = tensor.dim %0,    %c0 : tensor<?xf32>
  return %d0, %d1, %d2 : index, index, index
}

// -----

#map = affine_map<(d0) -> (d0)>

// Two dims that BOTH resolve to the compound (s0*s1): not bare symbols, so the
// Kind::Sym filter drops them and NEITHER is merged -- both survive. (A single
// compound dim would be skipped by the >=2 guard before the filter is reached,
// so two are needed to actually exercise the compound-skip path.)
// CHECK-LABEL: func.func @leave_compound
// CHECK-COUNT-2: tensor.dim
// CHECK-NOT: tensor.dim
// CHECK: return
func.func @leave_compound(
    %arg0: tensor<?xf32> {afir.symbolic_shape = "(s0*s1)"},
    %arg1: tensor<?xf32> {afir.symbolic_shape = "(s0*s1)"})
    -> (index, index)
    attributes {afir.dim_symbols = [
        {arg = 0 : i64, dim = 0 : i64, id = 0 : i64},
        {arg = 1 : i64, dim = 0 : i64, id = 1 : i64}]} {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>)
      attrs = {afir.symbolic_shapes = ["(s0*s1)"]} {
  ^bb0(%x: f32, %y: f32):
    linalg.yield %x : f32
  } -> tensor<?xf32>
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %d1 = tensor.dim %0,    %c0 : tensor<?xf32>
  return %d0, %d1 : index, index
}
