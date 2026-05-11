// RUN: not afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='materialization-mode=one-shot-bufferize' --ascend-compute-lower 2>&1 | FileCheck %s
// RUN: not afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate' --ascend-compute-lower 2>&1 | FileCheck %s --check-prefix=TARGET-AWARE

func.func @ordinary_elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
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
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK: error: ascend-compute-lower left a lowerable operation behind
// CHECK: "linalg.generic"

// TARGET-AWARE: error: ascend-compute-lower left a lowerable operation behind
// TARGET-AWARE: "linalg.generic"
