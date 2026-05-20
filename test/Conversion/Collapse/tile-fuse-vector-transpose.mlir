// Tail-axis transpose feeding an elementwise consumer — the "eliminate"
// template.  --linalg-fuse-elementwise-ops absorbs the named linalg.transpose
// into the relu generic's operand indexing map, so by tile-fuse time there is
// a single generic with a permuted operand map and no transpose op.  After
// tiling, the x operand is a row-strided subview (its tile dims are the
// iteration tile dims in transposed order); LinalgToAscendC then loads it with
// a per-row DataCopy and rearranges it with AscendC::Transpose before the relu
// (see examples/transpose-elementwise-e2e).

// RUN: afir-opt %s --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:   | FileCheck %s --check-prefix=ABSORB
// RUN: afir-opt %s --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:            --auto-fuse-tile-fuse | FileCheck %s --check-prefix=TILE

// The transpose op is gone; the relu reads x at (d0,d1)->(d1,d0).
// ABSORB-NOT: linalg.transpose
// ABSORB-DAG: #[[XMAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// ABSORB-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// ABSORB: linalg.generic
// ABSORB-SAME: indexing_maps = [#[[XMAP]], #[[ID]]]
// ABSORB-SAME: iterator_types = ["parallel", "parallel"]
// ABSORB: arith.maximumf

// After tiling: only the iteration row (output dim 0 = the block axis) is split;
// the other axis is fully loaded (§3.4 — non-ub parallel axes go whole, no
// `XBLOCK_SUB_n`).  x and the output are sliced with their tile dims in
// transposed order — x at offset [0, %iv] (x's leading layout dim is the
// iteration output row, left whole), the output at offset [%iv, 0].
// TILE-DAG: #[[XMAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// TILE-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// TILE: func.func @transpose_relu__v0(%{{.*}}: tensor<16x32xf16>, %{{.*}}: index {{{.*}}default_tile_size = 128{{.*}}}, %{{.*}}: index {{{.*}}default_tile_size = 16{{.*}}})
// TILE: tensor.extract_slice %arg0[%c0{{[0-9_]*}}, %[[IV:[a-zA-Z0-9_]+]]] [
// TILE: tensor.extract_slice %{{[^[]*}}[%[[IV]], %c0{{[0-9_]*}}] [
// TILE: linalg.generic {indexing_maps = [#[[XMAP]], #[[ID]]]
// TILE: arith.maximumf

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @transpose_relu(%x: tensor<16x32xf16>) -> tensor<32x16xf16> {
    %zero = arith.constant 0.0 : f16
    %t_init = tensor.empty() : tensor<32x16xf16>
    %t = linalg.transpose ins(%x : tensor<16x32xf16>)
                          outs(%t_init : tensor<32x16xf16>)
                          permutation = [1, 0]
    %r_init = tensor.empty() : tensor<32x16xf16>
    %out = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<32x16xf16>)
        outs(%r_init : tensor<32x16xf16>) {
    ^bb0(%in: f16, %o: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<32x16xf16>
    return %out : tensor<32x16xf16>
  }
}
