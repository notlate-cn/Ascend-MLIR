// RUN: ascend-mlir-opt %s --ascend-compute-lower | FileCheck %s --implicit-check-not=math.erf --implicit-check-not=math.tanh --implicit-check-not=math.sin --implicit-check-not=math.cos --implicit-check-not=math.ceil --implicit-check-not=math.floor --implicit-check-not=math.round --implicit-check-not=math.powf --implicit-check-not=linalg.generic

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @pyasc_math_unary_generic
// CHECK: ascendc.erf
// CHECK: ascendc.tanh
// CHECK: ascendc.sin
// CHECK: ascendc.cos
// CHECK: return
func.func @pyasc_math_unary_generic(%input: memref<4x8xf32>,
                                    %out: memref<4x8xf32>) {
  linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%input : memref<4x8xf32>)
      outs(%out : memref<4x8xf32>) {
    ^bb0(%x: f32, %out0: f32):
      %erf = math.erf %x : f32
      %tanh = math.tanh %erf : f32
      %sin = math.sin %tanh : f32
      %cos = math.cos %sin : f32
      linalg.yield %cos : f32
    }
  return
}

// CHECK-LABEL: func.func @pyasc_generic_math_generic
// CHECK: ascendc.ceil
// CHECK: ascendc.floor
// CHECK: ascendc.round
// CHECK: ascendc.power
// CHECK: return
func.func @pyasc_generic_math_generic(%input: memref<4x8xf32>,
                                      %exponent: memref<4x8xf32>,
                                      %out: memref<4x8xf32>) {
  linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%input, %exponent : memref<4x8xf32>, memref<4x8xf32>)
      outs(%out : memref<4x8xf32>) {
    ^bb0(%x: f32, %exp: f32, %out0: f32):
      %ceil = math.ceil %x : f32
      %floor = math.floor %ceil : f32
      %round = math.round %floor : f32
      %pow = math.powf %round, %exp : f32
      linalg.yield %pow : f32
    }
  return
}
