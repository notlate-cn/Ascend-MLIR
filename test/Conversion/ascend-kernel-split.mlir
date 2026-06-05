// RUN: sed -n '/\/\/ TRANSPOSE-BEGIN/,/\/\/ TRANSPOSE-END/p' %s | ascend-mlir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-kernel-split | FileCheck %s
// RUN: sed -n '/\/\/ CAPTURE-BEGIN/,/\/\/ CAPTURE-END/p' %s | ascend-mlir-opt --ascend-kernel-split | FileCheck %s --check-prefix=CAPTURE

// TRANSPOSE-BEGIN
func.func @transpose_chain(%arg0: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty0 = tensor.empty() : tensor<8x4xf16>
  %t0 = linalg.transpose ins(%arg0 : tensor<4x8xf16>)
      outs(%empty0 : tensor<8x4xf16>) permutation = [1, 0]
  %empty1 = tensor.empty() : tensor<4x8xf16>
  %t1 = linalg.transpose ins(%t0 : tensor<8x4xf16>)
      outs(%empty1 : tensor<4x8xf16>) permutation = [1, 0]
  return %t1 : tensor<4x8xf16>
}
// TRANSPOSE-END

// CHECK: module attributes
// CHECK-SAME: ascend.kernel_graph.edges
// CHECK-SAME: from = "kernel_0"
// CHECK-SAME: to = "kernel_1"
// CHECK-LABEL: func.func @kernel_0(
// CHECK-SAME: %{{.*}}: tensor<4x8xf16>
// CHECK-SAME: -> tensor<8x4xf16>
// CHECK-SAME: ascend.schedule.kernel_metadata
// CHECK-SAME: kernel = "kernel_0"
// CHECK: linalg.transpose
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK: return %{{.*}} : tensor<8x4xf16>
// CHECK-LABEL: func.func @kernel_1(
// CHECK-SAME: %{{.*}}: tensor<8x4xf16>
// CHECK-SAME: -> tensor<4x8xf16>
// CHECK-SAME: ascend.schedule.kernel_metadata
// CHECK-SAME: kernel = "kernel_1"
// CHECK: linalg.transpose
// CHECK-SAME: ascend.kernel = "kernel_1"
// CHECK: return %{{.*}} : tensor<4x8xf16>
// CHECK-NOT: func.func @transpose_chain

// CAPTURE-BEGIN
func.func @region_capture(%arg0: tensor<4xf32>) -> tensor<4xf32>
    attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<4xf32>
  %mid = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>)
    outs(%empty0 : tensor<4xf32>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<4xf32>

  %empty1 = tensor.empty() : tensor<4xf32>
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%empty1 : tensor<4xf32>)
    attrs = {
      ascend.kernel = "kernel_1",
      ascend.schedule.decision_id = "kernel_1.decision.0",
      ascend.schedule.schedule_contract = "generic_tiled_loop"
    } {
  ^bb0(%o: f32):
    %c0 = arith.constant 0 : index
    %v = tensor.extract %mid[%c0] : tensor<4xf32>
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %out : tensor<4xf32>
}
// CAPTURE-END

// CAPTURE-LABEL: func.func @kernel_0(
// CAPTURE-SAME: %{{.*}}: tensor<4xf32>
// CAPTURE-SAME: -> tensor<4xf32>
// CAPTURE-LABEL: func.func @kernel_1(
// CAPTURE-SAME: %[[MID:.*]]: tensor<4xf32>
// CAPTURE-SAME: -> tensor<4xf32>
// CAPTURE: tensor.extract %[[MID]]
// CAPTURE: return %{{.*}} : tensor<4xf32>
// CAPTURE-NOT: func.func @region_capture
