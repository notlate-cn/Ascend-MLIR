// RUN: afir-opt %s -split-input-file --afir-symbolize-shapes | FileCheck %s

// Pure elementwise: both args feed the same generic, so their shared dim is
// unified -- only one symbol per distinct arg dim survives in afir.dim_symbols.
// CHECK-LABEL: func.func @elementwise
// CHECK-SAME:  afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}]
// CHECK:       linalg.generic
// CHECK-SAME:  afir.symbolic_shapes = ["s0"]
func.func @elementwise(%a: tensor<?xf32>, %b: tensor<?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// collapse_shape of a 2-D dynamic tensor: result dim is the product of the two
// source-dim symbols.  The tensor.empty fed by arith.muli of the two dims gets
// the same expression.
// CHECK-LABEL: func.func @collapse
// CHECK-SAME:  afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}]
// CHECK:       tensor.collapse_shape
// CHECK-SAME:  afir.symbolic_shapes = ["(s0*s1)"]
// CHECK:       tensor.empty
// CHECK-SAME:  afir.symbolic_shapes = ["(s0*s1)"]
// CHECK:       linalg.generic
// CHECK-SAME:  afir.symbolic_shapes = ["(s0*s1)"]
func.func @collapse(%a: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %a, %c1 : tensor<?x?xf32>
  %coll = tensor.collapse_shape %a [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  %m = arith.muli %d0, %d1 : index
  %e = tensor.empty(%m) : tensor<?xf32>
  %add = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]}
      ins(%coll : tensor<?xf32>) outs(%e : tensor<?xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<?xf32>
  return %add : tensor<?xf32>
}

// -----

// reduce over d1: out[d0] = sum_d1 a[d0,d1] * b[d1].  b's dim and the init's dim
// are pinned to the same iteration dims as a's, so they're aliased away.
// CHECK-LABEL: func.func @reduce
// CHECK-SAME:  afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}, {arg = 0 : i64, dim = 1 : i64, id = 1 : i64}]
// CHECK:       linalg.generic
// CHECK-SAME:  afir.symbolic_shapes = ["s0"]
func.func @reduce(%a: tensor<?x?xf32>, %b: tensor<?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>],
      iterator_types = ["parallel","reduction"]}
      ins(%a, %b : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %m = arith.mulf %x, %y : f32
      %s = arith.addf %acc, %m : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// Mixed static/dynamic arg: the static dim is a constant SymExpr, the dynamic
// one a symbol.
// CHECK-LABEL: func.func @mixed
// CHECK-SAME:  afir.dim_symbols = [{arg = 0 : i64, dim = 1 : i64, id = 0 : i64}]
// CHECK:       linalg.generic
// CHECK-SAME:  afir.symbolic_shapes = ["4,s0"]
func.func @mixed(%a: tensor<4x?xf32>, %init: tensor<4x?xf32>) -> tensor<4x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]}
      ins(%a : tensor<4x?xf32>) outs(%init : tensor<4x?xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// -----

// Fully static function: nothing to symbolize, no attributes added.
// CHECK-LABEL: func.func @static
// CHECK-NOT:   afir.dim_symbols
// CHECK-NOT:   afir.symbolic_shapes
func.func @static(%a: tensor<8xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]}
      ins(%a : tensor<8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// Bail: a linalg.generic whose indexing map is not a projected permutation
// (here d0+d1 in an operand map) gets no afir.symbolic_shapes, and so does its
// consumer (which reads the unknown shape).
// CHECK-LABEL: func.func @bail_non_projperm
// CHECK-NOT:   afir.symbolic_shapes
func.func @bail_non_projperm(%a: tensor<?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0 + d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]}
      ins(%a : tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
