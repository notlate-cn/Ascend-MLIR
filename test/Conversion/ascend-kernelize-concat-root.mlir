// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

// CHECK: DependencyAnalysis
// CHECK: op_id = 2 op = "tensor.concat" participation = "analyze"
// CHECK: StructuralMarking
// CHECK: op_id = 2 merge_root = true
// CHECK: OpRoleClassification
// CHECK: op_id = 2 roles = ["Primary", "Vector", "LayoutTransform", "Merge"] op_role = "vector"
// CHECK: FusionCandidateAnalysis
// CHECK: primitive = "ConcatRootProducerFusion"
// CHECK-SAME: primary_ops = [2]
// CHECK-SAME: internal_ops = [0, 1, 2]
// CHECK: KernelPartition
// CHECK: kernel_pattern = "kernel_0" internal_ops = [0, 1, 2] primary_ops = [2]
// CHECK-NOT: kernel_pattern = "kernel_1"
// CHECK: tensor.concat
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK-SAME: ascend.primary = true

#broadcast_map = affine_map<(d0, d1) -> (d0)>
#full_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @concat_root_absorbs_elementwise_producers(
    %arg0: tensor<4xf16>,
    %arg1: tensor<4x8xf16>,
    %arg2: tensor<4xf16>,
    %arg3: tensor<4x8xf16>) -> tensor<8x8xf16> {
  %empty0 = tensor.empty() : tensor<4x8xf16>
  %0 = linalg.generic {
    indexing_maps = [#broadcast_map, #full_map, #full_map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4xf16>, tensor<4x8xf16>)
    outs(%empty0 : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %sum = arith.addf %x, %y : f16
    linalg.yield %sum : f16
  } -> tensor<4x8xf16>

  %empty1 = tensor.empty() : tensor<4x8xf16>
  %1 = linalg.generic {
    indexing_maps = [#broadcast_map, #full_map, #full_map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg2, %arg3 : tensor<4xf16>, tensor<4x8xf16>)
    outs(%empty1 : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %prod = arith.mulf %x, %y : f16
    linalg.yield %prod : f16
  } -> tensor<4x8xf16>

  %concat = tensor.concat dim(0) %0, %1
      : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<8x8xf16>
  return %concat : tensor<8x8xf16>
}
