// Multi-use transpose — the "preserve" template.  When a linalg.transpose has
// more than one consumer, --linalg-fuse-elementwise-ops cannot absorb it into a
// single downstream map, so it survives to tile-fuse as a standalone op.
// classifyAxes then applies AF's GenTransposeTilingGroup-style classification:
// for `out[d0,d1] = x[d1,d0]` the input-side divergent axis (d0, the output's
// outer dim) → X (its own inner tunable `XBLOCK_X_n`, never the block axis),
// the output-side axis (d1) → Y (the block axis: `XBLOCK` + `XBLOCK_SUB`).
//
// (The full preserve-template codegen — keeping the transpose intermediate
// on-chip as VECCALC, the 16-fractal split for >16 dims, and the preserve-vs-
// eliminate score — is follow-up work; this test pins the axis model.)

// RUN: afir-opt %s --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:            --canonicalize --vector-plan-tile-fuse | FileCheck %s

// The transpose op survives (yield-only generic, permuted operand map).
// CHECK-DAG: #[[XMAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: vector_plan.tiling_infos
// d0 (input-side divergent) → X: its own inner tunable, not the block axis.
// CHECK-SAME: name = "XBLOCK_X_0"
// d1 → Y: the block axis.
// CHECK-SAME: name = "XBLOCK"
// CHECK-SAME: name = "XBLOCK_SUB"
// CHECK: func.func @xp
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[XMAP]], #[[ID]]]
// CHECK: linalg.yield

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @xp(%x: tensor<16x32xf16>)
      -> (tensor<32x16xf16>, tensor<32x16xf16>) {
    %zero = arith.constant 0.0 : f16
    %two = arith.constant 2.0 : f16
    %ti = tensor.empty() : tensor<32x16xf16>
    %t = linalg.transpose ins(%x : tensor<16x32xf16>)
                          outs(%ti : tensor<32x16xf16>) permutation = [1, 0]
    %ai = tensor.empty() : tensor<32x16xf16>
    %a = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x16xf16>) outs(%ai : tensor<32x16xf16>) {
    ^bb0(%v: f16, %o: f16):
      %m = arith.maximumf %v, %zero : f16
      linalg.yield %m : f16
    } -> tensor<32x16xf16>
    %bi = tensor.empty() : tensor<32x16xf16>
    %b = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x16xf16>) outs(%bi : tensor<32x16xf16>) {
    ^bb0(%v: f16, %o: f16):
      %m = arith.mulf %v, %two : f16
      linalg.yield %m : f16
    } -> tensor<32x16xf16>
    return %a, %b : tensor<32x16xf16>, tensor<32x16xf16>
  }
}
