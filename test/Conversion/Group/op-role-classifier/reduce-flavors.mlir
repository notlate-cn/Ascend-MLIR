// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --auto-fuse-group-analysis '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.provenance.json

// Two independent reductions (rule 4 of OpRoleClassifier).
// reduce_sum: combiner = arith.addf  → "reduce_sum"
// reduce_max: combiner = arith.maximumf → "reduce_max"

#mapA = affine_map<(d0, d1) -> (d0, d1)>
#mapR = affine_map<(d0, d1) -> (d0)>

func.func @reduce_two(%inA: tensor<4x8xf16>, %iA: tensor<4xf16>,
                       %inB: tensor<4x8xf16>, %iB: tensor<4xf16>)
    -> (tensor<4xf16>, tensor<4xf16>) {
  %sum = linalg.generic {
    indexing_maps = [#mapA, #mapR],
    iterator_types = ["parallel", "reduction"]
  } ins(%inA : tensor<4x8xf16>) outs(%iA : tensor<4xf16>) {
  ^bb0(%a: f16, %acc: f16):
    %v = arith.addf %a, %acc : f16
    linalg.yield %v : f16
  } -> tensor<4xf16>

  %mx = linalg.generic {
    indexing_maps = [#mapA, #mapR],
    iterator_types = ["parallel", "reduction"]
  } ins(%inB : tensor<4x8xf16>) outs(%iB : tensor<4xf16>) {
  ^bb0(%a: f16, %acc: f16):
    %v = arith.maximumf %a, %acc : f16
    linalg.yield %v : f16
  } -> tensor<4xf16>

  return %sum, %mx : tensor<4xf16>, tensor<4xf16>
}

// CHECK: "op_role": "reduce_sum"
// CHECK: "op_role": "reduce_max"
