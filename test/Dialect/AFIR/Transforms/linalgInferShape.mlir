// RUN: afir-opt %s -split-input-file --linalg-infer-shape | FileCheck %s

// CHECK-LABEL:   func.func @generalize_matmul_buffer(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<16x?xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<?x32xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<16x32xf32>) -> tensor<16x32xf32> {
func.func @generalize_matmul_buffer(%A : tensor<16x?xf32>, %B: tensor<?x32xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // CHECK: %[[VAL_0:.*]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<16x?xf32>, tensor<?x32xf32>) outs(%[[ARG2]] : tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = linalg.matmul ins(%A, %B: tensor<16x?xf32>, tensor<?x32xf32>)
               outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL:   func.func @depthwise_conv_2d_nhwc_hwc(
// CHECK-SAME:      %[[ARG0:.*]]: memref<1x113x113x96xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<3x3x96xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<1x?x?x96xf32>) {
func.func @depthwise_conv_2d_nhwc_hwc(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<?x?x?x?xf32>) {
// CHECK: linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%[[ARG0]], %[[ARG1]] : memref<1x113x113x96xf32>, memref<3x3x96xf32>) outs(%[[ARG2]] : memref<1x?x?x96xf32>)
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<?x?x?x?xf32>)
  return
}