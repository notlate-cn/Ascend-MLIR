// AFIR Dialect representation of AscGraph (v2.0)
// Graph name: HashCopyAscGraph

// Indexing Maps
#map0 = affine_map<(d0, d1) -> (d0, d1)>

module attributes { afir.asc_graph_attr = #afir.asc_graph<axes = [<id=0,name="z0",axis_type=Original,size="20">, <id=1,name="z1",axis_type=Original,size="31">], type = Compute> } {

  func.func @HashCopyAscGraph(%arg0: tensor<20x31xf32>, %arg1: tensor<1x31xf32>, %arg2: i64 {afir.scalar_value="333"}) -> tensor<20x31xf32> {
%1 = afir.load %arg0 {
  indexing_maps = [#map0], ir_attr_def = {"offset" = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<20x31xf32> -> tensor<20x31xf32>
%3 = afir.load %arg1 {
  indexing_maps = [#map0], ir_attr_def = {"offset" = "0"}, outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<1x31xf32> -> tensor<1x31xf32>
%4 = afir.broadcast %3 {
  indexing_maps = [#map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<1x31xf32> -> tensor<20x31xf32>
%5 = afir.add %1, %4 {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%6 = afir.mul %5, %4 {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%7 = afir.sub %5, %6 {
  indexing_maps = [#map0, #map0, #map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : (tensor<20x31xf32>, tensor<20x31xf32>) -> tensor<20x31xf32>
%8 = afir.store %7 {
  indexing_maps = [#map0], outputs = [#afir.asc_tensor<tensor_id = -1, position = <gm>>]
} : tensor<20x31xf32> -> tensor<20x31xf32>
    return %8 : tensor<20x31xf32> 
  }
}
