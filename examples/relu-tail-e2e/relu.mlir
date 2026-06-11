// Elementwise relu, dynamic shape, sub-aligned tail E2E.
//   output[N] = max(input[N], 0)
func.func @relu(%a: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %a, %c0 : tensor<?xf32>
  %init = tensor.empty(%n) : tensor<?xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%a : tensor<?xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %zero = arith.constant 0.000000e+00 : f32
    %r = arith.maximumf %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
