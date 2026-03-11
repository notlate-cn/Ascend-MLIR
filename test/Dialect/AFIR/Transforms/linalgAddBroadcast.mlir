// RUN: afir-opt %s -split-input-file --linalg-mark --linalg-add-broadcast | FileCheck %s

func.func @unary_broadcast_addf(%A : tensor<16x32xf32>, %C : tensor<8x1x32xf32>, %B: tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<add>
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, 0, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
      ins(%A, %C : tensor<16x32xf32>, tensor<8x1x32xf32>)
      outs(%B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}

// CHECK-LABEL:   func.func @unary_broadcast_addf(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<16x32xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<8x1x32xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
// CHECK:           %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[}}0, 1], [2]] : tensor<8x1x32xf32> into tensor<8x32xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<8x16x32xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.broadcast ins(%[[ARG0]] : tensor<16x32xf32>) outs(%[[VAL_1]] : tensor<8x16x32xf32>) dimensions = [0]
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<8x16x32xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.broadcast ins(%[[VAL_0]] : tensor<8x32xf32>) outs(%[[VAL_3]] : tensor<8x16x32xf32>) dimensions = [1]
// CHECK:           %[[VAL_5:.*]] = linalg.elementwise kind=#linalg.elementwise_kind<add> {namedKind = 2 : index} ins(%[[VAL_2]], %[[VAL_4]] : tensor<8x16x32xf32>, tensor<8x16x32xf32>) outs(%[[ARG2]] : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
// CHECK:           return %[[VAL_5]] : tensor<8x16x32xf32>
// CHECK:         }

// -----

func.func @unary_broadcast_tanh(%A : tensor<1x16x32xf32>, %B: tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<tanh>
      indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
      ins(%A : tensor<1x16x32xf32>)
      outs(%B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}

// CHECK-LABEL:   func.func @unary_broadcast_tanh(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x16x32xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
// CHECK:           %[[VAL_0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1], [2]] : tensor<1x16x32xf32> into tensor<16x32xf32>
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<8x16x32xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.broadcast ins(%[[VAL_0]] : tensor<16x32xf32>) outs(%[[VAL_1]] : tensor<8x16x32xf32>) dimensions = [0]
// CHECK:           %[[VAL_3:.*]] = linalg.elementwise kind=#linalg.elementwise_kind<tanh> {namedKind = 2 : index} ins(%[[VAL_2]] : tensor<8x16x32xf32>) outs(%[[ARG1]] : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
// CHECK:           return %[[VAL_3]] : tensor<8x16x32xf32>
// CHECK:         }
