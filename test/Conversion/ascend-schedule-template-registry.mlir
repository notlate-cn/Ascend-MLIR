// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @vector_rank1(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
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

func.func @vector_rank2(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.mulf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

func.func @vector_rank3(%arg0: tensor<2x4x8xf16>, %arg1: tensor<2x4x8xf16>) -> tensor<2x4x8xf16> {
  %empty = tensor.empty() : tensor<2x4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<2x4x8xf16>, tensor<2x4x8xf16>)
    outs(%empty : tensor<2x4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<2x4x8xf16>
  return %out : tensor<2x4x8xf16>
}

func.func @reduction(%arg0: tensor<4x8xf16>) -> tensor<4xf16> {
  %empty = tensor.empty() : tensor<4xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<4x8xf16>)
    outs(%empty : tensor<4xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<4xf16>
  return %out : tensor<4xf16>
}

func.func @scalar_reduction(%arg0: tensor<8xf16>) -> tensor<f16> {
  %empty = tensor.empty() : tensor<f16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> ()>
    ],
    iterator_types = ["reduction"]
  } ins(%arg0 : tensor<8xf16>)
    outs(%empty : tensor<f16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<f16>
  return %out : tensor<f16>
}

func.func @matmul(%lhs: tensor<4x8xf16>, %rhs: tensor<8x16xf16>) -> tensor<4x16xf16> {
  %empty = tensor.empty() : tensor<4x16xf16>
  %out = linalg.matmul ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
                       outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %out : tensor<4x16xf16>
}

func.func @batch_matmul(%lhs: tensor<2x4x8xf16>,
                        %rhs: tensor<2x8x16xf16>) -> tensor<2x4x16xf16> {
  %empty = tensor.empty() : tensor<2x4x16xf16>
  %out = linalg.batch_matmul
      ins(%lhs, %rhs : tensor<2x4x8xf16>, tensor<2x8x16xf16>)
      outs(%empty : tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return %out : tensor<2x4x16xf16>
}

// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = vector_generic/single_tile_per_block
// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = vector_generic/single_tile_per_block
// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_2
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = vector_generic/single_tile_per_block
// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_3
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = reduction_static/single_tile_per_block
// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_4
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = reduction_static/single_tile_per_block
// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_5
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = cube_static_matmul/single_tile_per_block
// CHECK: TemplateRegistry:
// CHECK-NEXT:   kernel = kernel_6
// CHECK-NEXT:   matches = 1
// CHECK-NEXT:   template = cube_static_matmul/single_tile_per_block
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "vector_generic"
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "vector_generic"
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "vector_generic"
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "reduction_static"
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "reduction_static"
// CHECK: linalg.matmul
// CHECK-SAME: ascend.schedule.family = "cube_static_matmul"
// CHECK: linalg.batch_matmul
// CHECK-SAME: ascend.schedule.family = "cube_static_matmul"
