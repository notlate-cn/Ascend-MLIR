// RUN: afir-opt --convert-afir-to-ascir %s | FileCheck %s

// Test AFIR to ASC-IR conversion for add operation

#map = affine_map<(d0, d1) -> (d0, d1)>

// Test case 1: Basic tensor addition
// CHECK-LABEL: func.func @convert_add_basic
func.func @convert_add_basic(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: %0 = ascendc.tbuf : <veccalc>
  // CHECK: %1 = ascendc.tbuf.get_tensor %0 : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<4x4xf32>
  // CHECK: ascendc.add_l3 %1, %arg0, %arg1 : !ascendc.local_tensor<4x4xf32>, !ascendc.local_tensor<4x4xf32>, !ascendc.local_tensor<4x4xf32>
  %0 = afir.add %arg0, %arg1 {indexing_maps = [#map, #map, #map], outputs = [#afir.asc_tensor<vectorized_axis = [0, 1], vectorized_strides = [4, 1], tensor_id = 0, reuse_id=2, position = <vector_in, depth=2, is_double_buffer=true>, position_id=2>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
