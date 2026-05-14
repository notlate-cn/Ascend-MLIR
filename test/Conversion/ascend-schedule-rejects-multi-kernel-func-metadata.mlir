// RUN: not afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule 2>&1 | FileCheck %s

func.func @two_independent_kernels(%a: tensor<64xf16>,
                                   %b: tensor<64xf16>,
                                   %c: tensor<128xf16>,
                                   %d: tensor<128xf16>)
    -> (tensor<64xf16>, tensor<128xf16>) {
  %empty0 = tensor.empty() : tensor<64xf16>
  %out0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%a, %b : tensor<64xf16>, tensor<64xf16>)
    outs(%empty0 : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %sum = arith.addf %x, %y : f16
    linalg.yield %sum : f16
  } -> tensor<64xf16>

  %empty1 = tensor.empty() : tensor<128xf16>
  %out1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%c, %d : tensor<128xf16>, tensor<128xf16>)
    outs(%empty1 : tensor<128xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %product = arith.mulf %x, %y : f16
    linalg.yield %product : f16
  } -> tensor<128xf16>

  return %out0, %out1 : tensor<64xf16>, tensor<128xf16>
}

// CHECK: function contains multiple kernels with conflicting schedule metadata
