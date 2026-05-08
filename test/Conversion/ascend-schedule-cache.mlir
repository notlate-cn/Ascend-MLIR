// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @vector_rank2_a(%arg0: tensor<4x8xf16>,
                          %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
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
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

func.func @vector_rank2_b(%arg0: tensor<4x8xf16>,
                          %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
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

func.func @dynamic_vector(%arg0: tensor<?x8xf16>,
                          %arg1: tensor<?x8xf16>) -> tensor<?x8xf16> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x8xf16>
  %empty = tensor.empty(%d0) : tensor<?x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x8xf16>, tensor<?x8xf16>)
    outs(%empty : tensor<?x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<?x8xf16>
  return %out : tensor<?x8xf16>
}

func.func @reduction_prunes_full_tile(%arg0: tensor<2x2x2x2x2xf16>)
    -> tensor<2x2x2x2xf16> {
  %empty = tensor.empty() : tensor<2x2x2x2xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel",
                      "reduction"]
  } ins(%arg0 : tensor<2x2x2x2x2xf16>)
    outs(%empty : tensor<2x2x2x2xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %v = arith.addf %acc, %x : f16
    linalg.yield %v : f16
  } -> tensor<2x2x2x2xf16>
  return %out : tensor<2x2x2x2xf16>
}

// CHECK: ScheduleCache:
// CHECK-NEXT:   shape_bucket_lookups = 7
// CHECK-NEXT:   shape_bucket_misses = 3
// CHECK-NEXT:   tuning_lookups = 7
// CHECK-NEXT:   tuning_misses = 5
// CHECK-NEXT:   negative_cache_entries = 1
// CHECK-DAG:   shape_bucket_key = kernel_0|vector_static_2d|4x8
// CHECK-DAG:   shape_bucket_key = kernel_2|vector_static_2d|?x8
// CHECK-DAG:   tuning_result_key = kernel_0|vector_static_2d|single_tile_per_block|4x8|4x8
// CHECK-DAG:   tuning_result_key = kernel_0|vector_static_2d|single_tile_per_block|4x8|2x4
