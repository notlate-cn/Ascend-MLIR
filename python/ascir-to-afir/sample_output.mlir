// AFIR Dialect representation of AscGraph (v2.0)
// Graph name: HashCopyAscGraph

module attributes {asc_graph_attr = #afir.asc_graph<tiling_key = -1, axis = [#afir.axis<id = 0, name = "z0", axis_type = Original, bind_block = false, size = "20", align = "1", from = [], split_pair_other_id = -1>, #afir.axis<id = 1, name = "z1", axis_type = Original, bind_block = false, size = "31", align = "1", from = [], split_pair_other_id = -1>], type = COMPUTE, size_var = []>} {

  // Indexing Maps
  #map0 = affine_map<(d0, d1) -> (d0, d1)>

  func.func @HashCopyAscGraph() {
%0 = afir.data() {
  indexing_maps = [#map0], ir_attr_def = {"index" = 0 : i64}, outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : () -> tensor<20x31xf32>
%1 = afir.load(%0) {
  indexing_maps = [#map0], ir_attr_def = {"offset" = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<20x31xf32>) -> tensor<20x31xf32>
%2 = afir.data() {
  indexing_maps = [#map0], ir_attr_def = {"index" = 1 : i64}, outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : () -> tensor<1x31xf32>
%3 = afir.load(%2) {
  indexing_maps = [#map0], ir_attr_def = {"offset" = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<1x31xf32>) -> tensor<1x31xf32>
%4 = afir.broadcast(%3) {
  indexing_maps = [#map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<20x31xf32>) -> tensor<20x31xf32>
%5 = afir.add(%1, %4) {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%6 = afir.mul(%5, %4) {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%7 = afir.sub(%5, %6) {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%8 = afir.store(%7) {
  indexing_maps = [#map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<20x31xf32>) -> tensor<20x31xf32>
%9 = afir.output(%8) {
  indexing_maps = [#map0], ir_attr_def = {"index" = 0 : i64}, outputs = [#afir.asc_tensor<tensor_id = -1, position = GM>]
} : (tensor<20x31xf32>) -> tensor<20x31xf32>
    return
  }
}