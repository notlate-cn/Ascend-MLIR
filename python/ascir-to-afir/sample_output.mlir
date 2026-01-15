// AFIR Dialect representation of AscGraph (v2.0)
// Graph name: HashCopyAscGraph

// Indexing Maps
#map0 = affine_map<(d0, d1) -> (d0, d1)>

module attributes { afir.asc_graph_attr = #afir.asc_graph<axes = [<id=0,name="z0",axis_type=Original,size="20">, <id=1,name="z1",axis_type=Original,size="31">], type = Compute> } {

  func.func @HashCopyAscGraph(%arg0: tensor<20x31xf32>, %arg1: tensor<1x31xf32>) -> tensor<20x31xf32> {
%1 = afir.load %arg0 {
  indexing_maps = [#map0], ir_attr_def = {"offset" = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<20x31xf32> -> tensor<20x31xf32>
%2 = afir.load %arg1 {
  indexing_maps = [#map0], ir_attr_def = {"offset" = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<1x31xf32> -> tensor<1x31xf32>
%3 = afir.broadcast %2 {
  indexing_maps = [#map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<1x31xf32> -> tensor<20x31xf32>
%4 = afir.add %1, %3 {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%5 = afir.mul %4, %3 {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%6 = afir.sub %4, %5 {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%7 = afir.store %6 {
  indexing_maps = [#map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<20x31xf32> -> tensor<20x31xf32>
    return %7 : tensor<20x31xf32>
  }
}