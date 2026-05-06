// RUN: afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s --check-prefix=MLIR
// RUN: afir-opt %s --vector-plan-codegen 2>&1 | afir-translate -mlir-to-cann | FileCheck %s --check-prefix=CANN
//
// add+mul+relu 3-op elementwise fusion through --vector-plan-codegen.
// Input uses linalg.mul + linalg.add (named ops) + linalg.generic (relu)
// on dynamic 3D <?x?x?xf32> tensors. The pipeline inserts
// linalg-generalize-named-ops before TileFuse so all three become
// linalg.generic and fuse into a single tiled 1D loop.

// MLIR: func.func @add_mul_relu(
// MLIR-SAME: ascendc.aicore
// MLIR: emitasc.member {{.*}} "XBLOCK"
// MLIR: emitasc.member {{.*}} "XBLOCK_SUB"
// MLIR: emitasc.member {{.*}} "dim_arg0_0"
// MLIR: ascendc.get_block_idx
// MLIR: ascendc.mul_l2
// MLIR: ascendc.add_l2
// MLIR: ascendc.max_l2
// MLIR-NOT: linalg.mul
// MLIR-NOT: linalg.add
// MLIR-NOT: linalg.generic

// CANN: struct TilingData
// CANN: XBLOCK
// CANN: __aicore__ void add_mul_relu(
// CANN: Mul(
// CANN: Add(
// CANN: Max(

func.func @add_mul_relu(
    %a: tensor<?x?x?xf32>,
    %b: tensor<?x?x?xf32>,
    %c: tensor<?x?x?xf32>,
    %out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %a, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %a, %c2 : tensor<?x?x?xf32>
  %tmp_empty = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>

  // b * c
  %tmp = linalg.mul
    ins(%b, %c : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%tmp_empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  // a + b*c
  %add_empty = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %add = linalg.add
    ins(%a, %tmp : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%add_empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

  // relu = max(a + b*c, 0)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>,
                     affine_map<(d0,d1,d2)->(d0,d1,d2)>],
    iterator_types = ["parallel","parallel","parallel"]}
    ins(%add : tensor<?x?x?xf32>) outs(%out : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %unused: f32):
    %zero = arith.constant 0.0 : f32
    %r = arith.maximumf %in, %zero : f32
    linalg.yield %r : f32
  } -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}
