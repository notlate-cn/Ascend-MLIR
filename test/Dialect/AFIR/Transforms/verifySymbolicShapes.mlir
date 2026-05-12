// RUN: afir-opt %s -split-input-file -verify-diagnostics --afir-verify-symbolic-shapes

// Well-formed: dim_symbols + a matching symbolic_shapes -- no diagnostics.
func.func @ok(%a: tensor<?xf32>, %init: tensor<?xf32>) -> tensor<?xf32>
    attributes {afir.dim_symbols = [{arg = 0 : i64, dim = 0 : i64, id = 0 : i64}]} {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"], afir.symbolic_shapes = ["s0"]}
      ins(%a : tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Out-of-range symbol id (no afir.dim_symbols on the function).
func.func @bad_symbol(%a: tensor<?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{symbol id out of range}}
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"], afir.symbolic_shapes = ["s0"]}
      ins(%a : tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Wrong number of dims for the result rank.
func.func @bad_rank(%a: tensor<8xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @below {{type rank 1}}
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"], afir.symbolic_shapes = ["8,8"]}
      ins(%a : tensor<8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// Static dim disagrees with the constant in the symbolic shape.
func.func @bad_static(%a: tensor<8xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @below {{static dim is 8 but symbolic shape is 4}}
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"], afir.symbolic_shapes = ["4"]}
      ins(%a : tensor<8xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// Malformed SymExpr string.
func.func @bad_parse(%a: tensor<?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{failed to parse}}
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"], afir.symbolic_shapes = ["(s0+)"]}
      ins(%a : tensor<?xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%x: f32, %z: f32):
      linalg.yield %x : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
