func.func @relu(%a: tensor<1024xf32>) -> tensor<1024xf32> {
  %init = tensor.empty() : tensor<1024xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%a : tensor<1024xf32>) outs(%init : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %zero = arith.constant 0.000000e+00 : f32
    %r = arith.maximumf %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
