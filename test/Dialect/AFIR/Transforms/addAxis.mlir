// RUN: afir-opt --afir-add-axis %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2d = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1d = affine_map<(d0, d1, d2) -> (d2)>
module attributes {afir.asc_graph_attr = #afir.asc_graph<axes = [<id = 0, name = "z0", axis_type = Original, size = "20">, <id = 1, name = "z1", axis_type = Original, size = "31">, <id = 2, name = "z2", axis_type = Original, size = "32">], type = Compute>} {
  func.func @HashCopyAscGraph(%arg0: tensor<20x31x32xf32>, %arg1: tensor<31x32xf32>, %arg2:tensor<32xf32>) -> tensor<20x31x32xf32> {
    %0 = afir.load %arg0 {indexing_maps = [#map], ir_attr_def = {offset = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]} : tensor<20x31x32xf32> -> tensor<20x31x32xf32>
    %1 = afir.load %arg1 {indexing_maps = [#map2d], ir_attr_def = {offset = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]} : tensor<31x32xf32> -> tensor<31x32xf32>
    %2 = afir.load %arg2 {indexing_maps = [#map1d], ir_attr_def = {offset = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]} : tensor<32xf32> -> tensor<32xf32>
    %3 = afir.add %1, %2 {indexing_maps = [#map2d, #map1d, #map2d], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]} : (tensor<31x32xf32>, tensor<32xf32>) -> tensor<31x32xf32>
    %4 = afir.mul %3, %0 {indexing_maps = [#map2d, #map, #map], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]} : (tensor<31x32xf32>, tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %5 = afir.sub %3, %4 {indexing_maps = [#map2d, #map, #map], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]} : (tensor<31x32xf32>, tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
    %6 = afir.store %5 {indexing_maps = [#map], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]} : tensor<20x31x32xf32> -> tensor<20x31x32xf32>
    return %6 : tensor<20x31x32xf32>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (0, d1, d2)>
// CHECK-LABEL:   func.func @HashCopyAscGraph(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<20x31x32xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<1x31x32xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<1x1x32xf32>) -> tensor<20x31x32xf32> {
// CHECK:           %[[VAL_0:.*]] = afir.load %[[ARG0]] {indexing_maps = [#[[$ATTR_0]]], ir_attr_def = {offset = "0"}} : tensor<20x31x32xf32> -> tensor<20x31x32xf32>
// CHECK:           %[[VAL_1:.*]] = afir.load %[[ARG1]] {indexing_maps = [#[[$ATTR_0]]], ir_attr_def = {offset = "0"}} : tensor<1x31x32xf32> -> tensor<1x31x32xf32>
// CHECK:           %[[VAL_2:.*]] = afir.load %[[ARG2]] {indexing_maps = [#[[$ATTR_0]]], ir_attr_def = {offset = "0"}} : tensor<1x1x32xf32> -> tensor<1x1x32xf32>
// CHECK:           %[[VAL_3:.*]] = afir.broadcast %[[VAL_2]] {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_0]]], loop_axis = 0 : i32, outputs = [], tmp_buffers = []} : tensor<1x1x32xf32> -> tensor<1x31x32xf32>
// CHECK:           %[[VAL_4:.*]] = afir.add %[[VAL_1]], %[[VAL_3]] {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]]} : (tensor<1x31x32xf32>, tensor<1x31x32xf32>) -> tensor<1x31x32xf32>
// CHECK:           %[[VAL_5:.*]] = afir.broadcast %[[VAL_4]] {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_0]]], loop_axis = 0 : i32, outputs = [], tmp_buffers = []} : tensor<1x31x32xf32> -> tensor<20x31x32xf32>
// CHECK:           %[[VAL_6:.*]] = afir.mul %[[VAL_5]], %[[VAL_0]] {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]]} : (tensor<20x31x32xf32>, tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
// CHECK:           %[[VAL_7:.*]] = afir.broadcast %[[VAL_4]] {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_0]]], loop_axis = 0 : i32, outputs = [], tmp_buffers = []} : tensor<1x31x32xf32> -> tensor<20x31x32xf32>
// CHECK:           %[[VAL_8:.*]] = afir.sub %[[VAL_7]], %[[VAL_6]] {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]]} : (tensor<20x31x32xf32>, tensor<20x31x32xf32>) -> tensor<20x31x32xf32>
// CHECK:           %[[VAL_9:.*]] = afir.store %[[VAL_8]] {indexing_maps = [#[[$ATTR_0]]]} : tensor<20x31x32xf32> -> tensor<20x31x32xf32>
// CHECK:           return %[[VAL_9]] : tensor<20x31x32xf32>
// CHECK:         }