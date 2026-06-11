// RUN: afir-opt %s -split-input-file --linalg-to-ascendc | FileCheck %s

// torch.export lowers relu(x) to `arith.cmpf ugt, x, 0` + `arith.select`.
// --linalg-to-ascendc's SelectToMinMaxPattern normalizes it to arith.maximumf
// (before the body lowering, which knows max but not select+cmpf) so it isn't
// silently dropped.

// CHECK-LABEL: func.func @relu_select
// CHECK:       linalg.generic
// CHECK:         arith.maximumf
// CHECK-NOT:     arith.select
// CHECK-NOT:     arith.cmpf
func.func @relu_select(%a: tensor<16xf32>, %o: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0.0 : f32
  %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<16xf32>) outs(%o : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %cmp = arith.cmpf ugt, %in, %c0 : f32
    %s = arith.select %cmp, %in, %c0 : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %r : tensor<16xf32>
}

// -----

// `select(cmpf olt a b, a, b)` = min(a, b) -> arith.minimumf.

// CHECK-LABEL: func.func @min_select
// CHECK:         arith.minimumf
// CHECK-NOT:     arith.select
func.func @min_select(%a: tensor<16xf32>, %b: tensor<16xf32>, %o: tensor<16xf32>) -> tensor<16xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%o : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %cmp = arith.cmpf olt, %x, %y : f32
    %s = arith.select %cmp, %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %r : tensor<16xf32>
}

// -----

// `select(cmpf oeq a b, a, b)` is NOT a min/max -> left alone.

// CHECK-LABEL: func.func @eq_select
// CHECK:         arith.select
func.func @eq_select(%a: tensor<16xf32>, %b: tensor<16xf32>, %o: tensor<16xf32>) -> tensor<16xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%o : tensor<16xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %cmp = arith.cmpf oeq, %x, %y : f32
    %s = arith.select %cmp, %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<16xf32>
  return %r : tensor<16xf32>
}
