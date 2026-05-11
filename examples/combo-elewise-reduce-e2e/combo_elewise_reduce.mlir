// Combo end-to-end: elementwise add + reduce(sum, axis=2), f32, 3D.
//   y[d0,d1,d2] = a[d0,d1,d2] + b[d0,d1,d2]
//   out[d0,d1]  = sum_{d2}( y[d0,d1,d2] )
//
// %init is the reduce accumulator; in the CANN signature it becomes the
// output slot.  The elementwise add's scratch tensor is materialized via
// tensor.empty inside the function so it does NOT add a kernel argument.
// After --linalg-fuse-elementwise-ops the add gets inlined into the reduce
// body, the empty becomes dead, and cann.num_inputs = 2 (a, b).

#map_full   = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @combo_elewise_reduce(
      %a: tensor<?x?x?xf32>,
      %b: tensor<?x?x?xf32>,
      %init: tensor<?x?xf32>) -> tensor<?x?xf32> {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %a, %c0 : tensor<?x?x?xf32>
    %d1 = tensor.dim %a, %c1 : tensor<?x?x?xf32>
    %d2 = tensor.dim %a, %c2 : tensor<?x?x?xf32>
    %ey = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>

    %y = linalg.generic {
        indexing_maps = [#map_full, #map_full, #map_full],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%ey : tensor<?x?x?xf32>) {
    ^bb0(%av: f32, %bv: f32, %_: f32):
      %s = arith.addf %av, %bv : f32
      linalg.yield %s : f32
    } -> tensor<?x?x?xf32>

    %out = linalg.generic {
        indexing_maps = [#map_full, #map_reduce],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%y : tensor<?x?x?xf32>)
        outs(%init : tensor<?x?xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %v = arith.addf %acc, %in : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>

    return %out : tensor<?x?xf32>
  }
}
