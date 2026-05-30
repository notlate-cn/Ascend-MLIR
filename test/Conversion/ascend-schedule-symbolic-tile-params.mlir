// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @symbolic_vector(%arg0: tensor<70x128xf16>,
                           %arg1: tensor<70x128xf16>) -> tensor<70x128xf16> {
  %empty = tensor.empty() : tensor<70x128xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<70x128xf16>, tensor<70x128xf16>)
    outs(%empty : tensor<70x128xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<70x128xf16>
  return %out : tensor<70x128xf16>
}

func.func @symbolic_reduction(%arg0: tensor<70x128xf16>) -> tensor<70xf16> {
  %empty = tensor.empty() : tensor<70xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<70x128xf16>)
    outs(%empty : tensor<70xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<70xf16>
  return %out : tensor<70xf16>
}

func.func @symbolic_cube(%lhs: tensor<70x64xf16>,
                         %rhs: tensor<64x128xf16>) -> tensor<70x128xf16> {
  %empty = tensor.empty() : tensor<70x128xf16>
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<70x64xf16>, tensor<64x128xf16>)
      outs(%empty : tensor<70x128xf16>) -> tensor<70x128xf16>
  return %out : tensor<70x128xf16>
}

// CHECK: ScheduleDecisionSet:
// CHECK:   tile_params = [name=TB_M axis=0 binding=runtime
// CHECK-SAME: primitive_uses=[data_copy,vector_compute,write_back]
// CHECK: ScheduleDecisionSet:
// CHECK:   tile_params = [name=TB_M axis=0 binding=runtime
// CHECK-SAME: [name=TB_N axis=1 binding=extent
// CHECK-SAME: primitive_uses=[reduction]
// CHECK: ScheduleDecisionSet:
// CHECK:   tile_params = [name=TB_M axis=0 binding=runtime
// CHECK-SAME: [name=t_K axis=2 binding=extent
// CHECK-SAME: primitive_uses=[reduction,cube_k]
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.tile_binding = "symbolic"
// CHECK-SAME: ascend.schedule.tile_params = [
// CHECK: linalg.matmul
// CHECK-SAME: ascend.schedule.tile_binding = "symbolic"
// CHECK-SAME: ascend.schedule.tile_params = [
