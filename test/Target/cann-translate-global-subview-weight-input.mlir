// RUN: ascend-mlir-opt --ascend-prepare-for-emit --ascend-canonicalize-cann-signature %s | ascend-mlir-translate -mlir-to-cann - | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void global_subview_weight(
// CHECK: GM_ADDR
// CHECK: GM_ADDR
// CHECK: GM_ADDR
// CHECK: TilingData
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ float*>(

module {
  memref.global "private" constant @weights : memref<128xf32> =
      dense<1.000000e+00> {alignment = 64 : i64}

  func.func @global_subview_weight(
      %input: memref<128xf32>,
      %output: memref<128xf32>
  ) attributes {ascendc.aicore, ascendc.global, ascendc.kernel_kind = "vec"} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %weights = memref.get_global @weights : memref<128xf32>
    %slice = memref.subview %weights[%c0] [%c128] [1]
        : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>>
    %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
    ascendc.global_tensor.set_global_buffer %gt, %slice
        : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[1], offset: ?>>
    func.return
  }
}
