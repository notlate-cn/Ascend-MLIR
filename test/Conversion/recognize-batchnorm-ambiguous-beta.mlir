// RUN: afir-opt --recognize-batchnorm %s | FileCheck %s

// A BatchNorm-shaped region (inference form: (x-mean)/sqrt(var+eps)*gamma+beta)
// where the gamma-scaled value %scaled is consumed by TWO addf generics with
// rank-1 operands.  recognize-batchnorm picks beta as "the other input of the
// (first) addf user of %scaled"; with two candidates it cannot tell which add
// is the real affine bias, and folding would silently capture the wrong one.
// The pass must refuse to fold an ambiguous match.

#r1  = affine_map<(c) -> (c)>
#id2 = affine_map<(n, c) -> (n, c)>
#bc  = affine_map<(n, c) -> (0, c)>

func.func @bn_ambiguous_beta(%x: tensor<2x4xf32>, %mean: tensor<4xf32>,
    %var: tensor<4xf32>, %gamma: tensor<4xf32>, %beta: tensor<4xf32>,
    %beta2: tensor<4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %eps = arith.constant 1.000000e-05 : f32
  %one = arith.constant 1.000000e+00 : f32
  %e1 = tensor.empty() : tensor<4xf32>
  %e2 = tensor.empty() : tensor<2x4xf32>

  %addeps = linalg.generic {indexing_maps = [#r1, #r1], iterator_types = ["parallel"]}
            ins(%var : tensor<4xf32>) outs(%e1 : tensor<4xf32>) {
  ^bb0(%a: f32, %o: f32): %s = arith.addf %a, %eps : f32
    linalg.yield %s : f32 } -> tensor<4xf32>
  %sqrt = linalg.generic {indexing_maps = [#r1, #r1], iterator_types = ["parallel"]}
          ins(%addeps : tensor<4xf32>) outs(%e1 : tensor<4xf32>) {
  ^bb0(%a: f32, %o: f32): %r = math.sqrt %a : f32
    linalg.yield %r : f32 } -> tensor<4xf32>
  %inv = linalg.generic {indexing_maps = [#r1, #r1], iterator_types = ["parallel"]}
         ins(%sqrt : tensor<4xf32>) outs(%e1 : tensor<4xf32>) {
  ^bb0(%a: f32, %o: f32): %d = arith.divf %one, %a : f32
    linalg.yield %d : f32 } -> tensor<4xf32>

  %inv_e   = tensor.expand_shape %inv   [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
  %mean_e  = tensor.expand_shape %mean  [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
  %gamma_e = tensor.expand_shape %gamma [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
  %beta_e  = tensor.expand_shape %beta  [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
  %beta2_e = tensor.expand_shape %beta2 [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>

  %cen = linalg.generic {indexing_maps = [#id2, #bc, #id2], iterator_types = ["parallel","parallel"]}
         ins(%x, %mean_e : tensor<2x4xf32>, tensor<1x4xf32>) outs(%e2 : tensor<2x4xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %s = arith.subf %a, %b : f32
    linalg.yield %s : f32 } -> tensor<2x4xf32>
  %norm = linalg.generic {indexing_maps = [#id2, #bc, #id2], iterator_types = ["parallel","parallel"]}
          ins(%cen, %inv_e : tensor<2x4xf32>, tensor<1x4xf32>) outs(%e2 : tensor<2x4xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32 } -> tensor<2x4xf32>
  %scaled = linalg.generic {indexing_maps = [#id2, #bc, #id2], iterator_types = ["parallel","parallel"]}
            ins(%norm, %gamma_e : tensor<2x4xf32>, tensor<1x4xf32>) outs(%e2 : tensor<2x4xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32 } -> tensor<2x4xf32>

  // two competing rank-1 addf users of %scaled -> ambiguous beta
  %out = linalg.generic {indexing_maps = [#id2, #bc, #id2], iterator_types = ["parallel","parallel"]}
         ins(%scaled, %beta_e : tensor<2x4xf32>, tensor<1x4xf32>) outs(%e2 : tensor<2x4xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %s = arith.addf %a, %b : f32
    linalg.yield %s : f32 } -> tensor<2x4xf32>
  %out2 = linalg.generic {indexing_maps = [#id2, #bc, #id2], iterator_types = ["parallel","parallel"]}
          ins(%scaled, %beta2_e : tensor<2x4xf32>, tensor<1x4xf32>) outs(%e2 : tensor<2x4xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %s = arith.addf %a, %b : f32
    linalg.yield %s : f32 } -> tensor<2x4xf32>
  return %out, %out2 : tensor<2x4xf32>, tensor<2x4xf32>
}

// Ambiguous beta -> must NOT fold.
// CHECK-LABEL: func.func @bn_ambiguous_beta
// CHECK: math.sqrt
// CHECK-NOT: aclnn.kind = "batch_norm"
