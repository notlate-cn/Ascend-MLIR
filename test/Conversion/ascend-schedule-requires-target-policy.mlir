// RUN: not ascend-mlir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule 2>&1 | FileCheck %s

func.func @requires_explicit_target_policy(%arg0: tensor<64xf16>,
                                           %arg1: tensor<64xf16>)
    -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %old: f16):
    %sum = arith.addf %x, %y : f16
    linalg.yield %sum : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK: ascend-schedule requires explicit target-tile-policy
// CHECK: target-tile-policy=legacy-default
// CHECK: target-tile-policy=target-aware
