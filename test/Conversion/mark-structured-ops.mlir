// RUN: afir-opt --mark-structured-ops %s | FileCheck %s

// ============================================================
// Test 1: index_select pattern (column gather) -> gather_dim = 1
// ============================================================
// CHECK-LABEL: func.func @test_index_select
// CHECK: linalg.generic
// CHECK-SAME: gather_dim = 1
// CHECK-NOT: embedding_dim
func.func @test_index_select(
    %data    : tensor<4x8xf16>,
    %indices : tensor<3xi64>
) -> tensor<4x3xf16> {
  %empty = tensor.empty() : tensor<4x3xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%indices : tensor<3xi64>)
    outs(%empty : tensor<4x3xf16>) {
  ^bb0(%idx: i64, %o: f16):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %ic = arith.index_cast %idx : i64 to index
    %v = tensor.extract %data[%i, %ic] : tensor<4x8xf16>
    linalg.yield %v : f16
  } -> tensor<4x3xf16>
  return %out : tensor<4x3xf16>
}

// ============================================================
// Test 2: embedding pattern (row gather) -> embedding_dim = 0
// ============================================================
// CHECK-LABEL: func.func @test_embedding
// CHECK: linalg.generic
// CHECK-SAME: embedding_dim = 0
// CHECK-NOT: gather_dim
func.func @test_embedding(
    %weight  : tensor<16x8xf16>,
    %indices : tensor<4xi64>
) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%indices : tensor<4xi64>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%idx: i64, %o: f16):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %ic = arith.index_cast %idx : i64 to index
    %v = tensor.extract %weight[%ic, %j] : tensor<16x8xf16>
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

// ============================================================
// Test 3: plain elementwise (no tensor.extract) -> no attribute
// ============================================================
// CHECK-LABEL: func.func @test_elementwise
// CHECK: linalg.generic
// CHECK-NOT: gather_dim
// CHECK-NOT: embedding_dim
func.func @test_elementwise(
    %a : tensor<4x8xf16>,
    %b : tensor<4x8xf16>
) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}
