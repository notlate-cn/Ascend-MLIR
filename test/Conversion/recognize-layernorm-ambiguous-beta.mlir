// RUN: afir-opt --recognize-layernorm %s | FileCheck %s

// A LayerNorm-shaped region where the gamma-scaled value `%sc` is consumed by
// TWO rank-1 addf generics (a real beta add AND a second rank-1 bias/residual
// add).  recognize-layernorm identifies beta as "the other input of the (first)
// addf user of %sc" -- with two candidates it cannot tell which add is the real
// affine bias.  Folding to @__aclnn_layer_norm would silently pick one (the use
// order decides), capturing the wrong tensor as beta and dropping the other op.
//
// The pass must refuse to fold an ambiguous match rather than guess: no
// @__aclnn_layer_norm, and the rsqrt/adds stay for the generic path.

#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#rd = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#g  = affine_map<(d0, d1, d2) -> (d2)>
#mb = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @ln_ambiguous_beta(%x: tensor<2x4x8xf32>, %gamma: tensor<8xf32>,
                             %beta: tensor<8xf32>, %beta2: tensor<8xf32>)
    -> (tensor<2x4x8xf32>, tensor<2x4x8xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %n   = arith.constant 8.000000e+00 : f32
  %eps = arith.constant 9.999999747378752e-06 : f32
  %e3  = tensor.empty() : tensor<2x4x8xf32>
  %e1  = tensor.empty() : tensor<2x4x1xf32>
  %f1  = linalg.fill ins(%cst : f32) outs(%e1 : tensor<2x4x1xf32>) -> tensor<2x4x1xf32>

  %sum = linalg.generic {indexing_maps = [#id, #rd], iterator_types = ["parallel","parallel","reduction"]}
         ins(%x : tensor<2x4x8xf32>) outs(%f1 : tensor<2x4x1xf32>) {
  ^bb0(%a: f32, %o: f32): %s = arith.addf %a, %o : f32
    linalg.yield %s : f32 } -> tensor<2x4x1xf32>
  %mean = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel","parallel"]}
          ins(%sum : tensor<2x4x1xf32>) outs(%e1 : tensor<2x4x1xf32>) {
  ^bb0(%a: f32, %o: f32): %d = arith.divf %a, %n : f32
    linalg.yield %d : f32 } -> tensor<2x4x1xf32>
  %meanc = tensor.collapse_shape %mean [[0],[1,2]] : tensor<2x4x1xf32> into tensor<2x4xf32>
  %meanb = linalg.generic {indexing_maps = [#mb, #id], iterator_types = ["parallel","parallel","parallel"]}
           ins(%meanc : tensor<2x4xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %o: f32): linalg.yield %a : f32 } -> tensor<2x4x8xf32>
  %cen = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel","parallel","parallel"]}
         ins(%x, %meanb : tensor<2x4x8xf32>, tensor<2x4x8xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %s = arith.subf %a, %b : f32
    linalg.yield %s : f32 } -> tensor<2x4x8xf32>
  %sq = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel","parallel","parallel"]}
        ins(%cen, %cen : tensor<2x4x8xf32>, tensor<2x4x8xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32 } -> tensor<2x4x8xf32>
  %vsum = linalg.generic {indexing_maps = [#id, #rd], iterator_types = ["parallel","parallel","reduction"]}
          ins(%sq : tensor<2x4x8xf32>) outs(%f1 : tensor<2x4x1xf32>) {
  ^bb0(%a: f32, %o: f32): %s = arith.addf %a, %o : f32
    linalg.yield %s : f32 } -> tensor<2x4x1xf32>
  %var = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel","parallel"]}
         ins(%vsum : tensor<2x4x1xf32>) outs(%e1 : tensor<2x4x1xf32>) {
  ^bb0(%a: f32, %o: f32): %d = arith.divf %a, %n : f32
    linalg.yield %d : f32 } -> tensor<2x4x1xf32>
  %vare = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel","parallel"]}
          ins(%var : tensor<2x4x1xf32>) outs(%e1 : tensor<2x4x1xf32>) {
  ^bb0(%a: f32, %o: f32): %s = arith.addf %a, %eps : f32
    linalg.yield %s : f32 } -> tensor<2x4x1xf32>
  %rstd = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel","parallel","parallel"]}
          ins(%vare : tensor<2x4x1xf32>) outs(%e1 : tensor<2x4x1xf32>) {
  ^bb0(%a: f32, %o: f32): %r = math.rsqrt %a : f32
    linalg.yield %r : f32 } -> tensor<2x4x1xf32>
  %rstdc = tensor.collapse_shape %rstd [[0],[1,2]] : tensor<2x4x1xf32> into tensor<2x4xf32>
  %rstdb = linalg.generic {indexing_maps = [#mb, #id], iterator_types = ["parallel","parallel","parallel"]}
           ins(%rstdc : tensor<2x4xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %o: f32): linalg.yield %a : f32 } -> tensor<2x4x8xf32>
  %norm = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel","parallel","parallel"]}
          ins(%cen, %rstdb : tensor<2x4x8xf32>, tensor<2x4x8xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32 } -> tensor<2x4x8xf32>
  %sc = linalg.generic {indexing_maps = [#id, #g, #id], iterator_types = ["parallel","parallel","parallel"]}
        ins(%norm, %gamma : tensor<2x4x8xf32>, tensor<8xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32 } -> tensor<2x4x8xf32>
  // Two competing rank-1 addf users of %sc -> ambiguous beta.
  %out = linalg.generic {indexing_maps = [#id, #g, #id], iterator_types = ["parallel","parallel","parallel"]}
         ins(%sc, %beta : tensor<2x4x8xf32>, tensor<8xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %s = arith.addf %a, %b : f32
    linalg.yield %s : f32 } -> tensor<2x4x8xf32>
  %out2 = linalg.generic {indexing_maps = [#id, #g, #id], iterator_types = ["parallel","parallel","parallel"]}
          ins(%sc, %beta2 : tensor<2x4x8xf32>, tensor<8xf32>) outs(%e3 : tensor<2x4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32): %s = arith.addf %a, %b : f32
    linalg.yield %s : f32 } -> tensor<2x4x8xf32>
  return %out, %out2 : tensor<2x4x8xf32>, tensor<2x4x8xf32>
}

// Ambiguous beta -> must NOT fold.
// CHECK-LABEL: func.func @ln_ambiguous_beta
// CHECK: math.rsqrt
// CHECK-NOT: __aclnn_layer_norm
