// RUN: afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline %s | FileCheck %s

// Regression: group analysis must not create a cyclic group when the cross-group
// dependency is routed through non-linalg glue (tensor.collapse_shape /
// expand_shape).  Here op_a and op_b share input %x (horizontal-fusion
// candidate), but op_c sits between them via reshapes: a -> collapse -> c ->
// expand -> b.  The cycle-prevention check used to look only at direct linalg
// operands, so it was blind to the glue-routed a->c and c->b edges, merged a+b,
// and produced a cyclic, non-outlinable group (the transformer-encoder crash).
// With glue-aware cycle detection, a and b are NOT merged and outlining succeeds.

#map  = affine_map<(d0, d1) -> (d0, d1)>
#flat = affine_map<(d0) -> (d0)>

func.func @glue_cycle(%x: tensor<4x4xf16>, %y: tensor<4x4xf16>,
                      %i1: tensor<4x4xf16>, %i2: tensor<4x4xf16>,
                      %ic: tensor<16xf16>)
    -> (tensor<4x4xf16>, tensor<4x4xf16>) {
  %ra = linalg.generic {indexing_maps = [#map, #map, #map],
                        iterator_types = ["parallel", "parallel"]}
        ins(%x, %y : tensor<4x4xf16>, tensor<4x4xf16>) outs(%i1 : tensor<4x4xf16>) {
  ^bb0(%a: f16, %b: f16, %o: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>

  %g = tensor.collapse_shape %ra [[0, 1]] : tensor<4x4xf16> into tensor<16xf16>

  %rc = linalg.generic {indexing_maps = [#flat, #flat], iterator_types = ["parallel"]}
        ins(%g : tensor<16xf16>) outs(%ic : tensor<16xf16>) {
  ^bb0(%a: f16, %o: f16):
    %v = arith.negf %a : f16
    linalg.yield %v : f16
  } -> tensor<16xf16>

  %re = tensor.expand_shape %rc [[0, 1]] output_shape [4, 4]
        : tensor<16xf16> into tensor<4x4xf16>

  %rb = linalg.generic {indexing_maps = [#map, #map, #map],
                        iterator_types = ["parallel", "parallel"]}
        ins(%x, %re : tensor<4x4xf16>, tensor<4x4xf16>) outs(%i2 : tensor<4x4xf16>) {
  ^bb0(%a: f16, %b: f16, %o: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<4x4xf16>

  return %ra, %rb : tensor<4x4xf16>, tensor<4x4xf16>
}

// Outlining must SUCCEED (no "not contiguous" cyclic-group error) and emit
// private kernel funcs plus the rewritten coordinator.
// CHECK: func.func private @kernel_group
// CHECK: func.func @glue_cycle
