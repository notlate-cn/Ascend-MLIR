// RUN: afir-opt --afir-shape-inference %s | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1d = affine_map<(d0) -> (d0)>
#map2d = affine_map<(d0, d1) -> (d0, d1)>
#map3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @HashCopyAscGraph(%arg0: tensor<4x1x31xf32>, %arg1: tensor<1x30x1xf32>) -> tensor<?x?x?xf32> {
    // CHECK: -> tensor<4x30x31xf32>
    %2 = afir.add %arg0, %arg1 : (tensor<4x1x31xf32>, tensor<1x30x1xf32>) -> tensor<?x?x?xf32>
    %3 = afir.mul %2, %arg1 : (tensor<?x?x?xf32>, tensor<1x30x1xf32>) -> tensor<?x?x?xf32>
    %4 = afir.sub %arg1, %3 : (tensor<1x30x1xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %4 : tensor<?x?x?xf32>
  }

func.func @test_clip_by_value(%arg0: tensor<4x1x1xf32>, %arg1: tensor<1x20x1xf32>, %arg2: tensor<1x1x32xf32>) -> tensor<?x?x?xf32> {
    // CHECK: -> tensor<4x20x32xf32>
    %0 = afir.clip_by_value %arg0, %arg1, %arg2 : (tensor<4x1x1xf32>, tensor<1x20x1xf32>, tensor<1x1x32xf32>) -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
}

func.func @test_concat(%arg0: tensor<2x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<?x?xf32> {
    // CHECK: -> tensor<9x4xf32>
  %0 = afir.concat %arg0, %arg1, %arg2 {indexing_maps = [#map2d, #map2d, #map2d, #map2d], concat_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4xf32>, tensor<3x4xf32>, tensor<4x4xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

func.func @test_two_concat(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<?x?xf32> {
    // CHECK: -> tensor<9x4xf32>
  %0 = afir.concat %arg0, %arg1 {indexing_maps = [#map2d, #map2d, #map2d, #map2d], concat_axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<2x4xf32>, tensor<2x4xf32> -> tensor<?x?xf32>
  %1 = afir.concat %0, %arg2 {indexing_maps = [#map2d, #map2d, #map2d, #map2d], concat_axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<?x?xf32>, tensor<4x4xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

func.func @test_max_reduce_f16(%arg0: tensor<4x5xf16>) -> (tensor<?xf16>, tensor<?xf16>) {
    // CHECK: -> tensor<5xf16>
  %0 = afir.max %arg0 {indexing_maps = [#map2d, #map1d], axis = 0 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x5xf16> -> tensor<?xf16>
  // CHECK: -> tensor<4xf16>
  %1 = afir.max %arg0 {indexing_maps = [#map2d, #map1d], axis = 1 : i32, outputs = [#afir.asc_tensor<vectorized_axis = [0], vectorized_strides = [1], tensor_id = 0, reuse_id = -1, position = <vector_in, depth = 0, is_double_buffer = false>, position_id = 0>]} : tensor<4x5xf16> -> tensor<?xf16>
  return %0, %1 : tensor<?xf16>, tensor<?xf16>
}