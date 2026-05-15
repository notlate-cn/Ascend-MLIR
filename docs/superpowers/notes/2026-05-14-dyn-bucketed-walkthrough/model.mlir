// dyn-bucketed-e2e
//
// 3D fully-dynamic compute graph with two independent chains, designed to
// trigger bucketing (GroupOutline) and run end-to-end without aclnn.
//
// Each chain has 3–4 disjoint inputs, so the horizontal-fuse limit
// (max-horizontal-extra-inputs=4) is exceeded and GroupAnalysis emits TWO
// groups; GroupOutline then outlines each into its own kernel func.
//
// Chain 0 (elementwise + reduce-axis-2):
//     out0[d0,d1] = init0[d0,d1] + sum_{d2}( (a + b) * c + d )
// Chain 1 (elementwise + reduce-axis-2):
//     out1[d0,d1] = init1[d0,d1] + sum_{d2}( (e + f) * g )
//
// Both chains carry a reduction iterator so that --linalg-fold-unit-extent-dims
// keeps the init operand bound to the function argument (i.e. `outs(%initN)`
// survives).  Otherwise the pass would move init into `ins` and synthesize a
// tensor.empty in the coordinator, which emitNetworkJson rejects (the
// coordinator only accepts tensor.cast / reshape / func.call / func.return).
//
// Each chain is a single linalg.generic to avoid intermediate tensor.empty
// scratch tensors leaking into the coordinator after fusion / folding.
//
// No aclnn-routable ops are used; the entire pipeline goes linalg → AscendC.

#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @bucketed_dyn(
      %a: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>,
      %c: tensor<?x?x?xf32>, %d: tensor<?x?x?xf32>,
      %e: tensor<?x?x?xf32>, %f: tensor<?x?x?xf32>,
      %g: tensor<?x?x?xf32>,
      %init0: tensor<?x?xf32>, %init1: tensor<?x?xf32>)
      -> (tensor<?x?xf32>, tensor<?x?xf32>) {

    // ────────── Chain 0: out0 = init0 + sum_d2( (a+b)*c - d ) ──────────
    %out0 = linalg.generic {
        indexing_maps = [#map3, #map3, #map3, #map3, #map2],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b, %c, %d :
            tensor<?x?x?xf32>, tensor<?x?x?xf32>,
            tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%init0 : tensor<?x?xf32>) {
    ^bb0(%va: f32, %vb: f32, %vc: f32, %vd: f32, %acc: f32):
      %p1 = arith.addf %va, %vb : f32
      %p2 = arith.mulf %p1, %vc : f32
      %p3 = arith.addf %p2, %vd : f32
      %p4 = arith.addf %acc, %p3 : f32
      linalg.yield %p4 : f32
    } -> tensor<?x?xf32>

    // ────────── Chain 1: out1 = init1 + sum_d2( (e+f)*g ) ──────────
    %out1 = linalg.generic {
        indexing_maps = [#map3, #map3, #map3, #map2],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%e, %f, %g :
            tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%init1 : tensor<?x?xf32>) {
    ^bb0(%ve: f32, %vf: f32, %vg: f32, %acc: f32):
      %q1 = arith.addf %ve, %vf : f32
      %q2 = arith.mulf %q1, %vg : f32
      %q3 = arith.addf %acc, %q2 : f32
      linalg.yield %q3 : f32
    } -> tensor<?x?xf32>

    return %out0, %out1 : tensor<?x?xf32>, tensor<?x?xf32>
  }
}
