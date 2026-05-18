// RUN: split-file %s %t
// RUN: python3 %S/check_ascend_queue_lifetime.py %t/good.mlir | FileCheck %s --check-prefix=GOOD
// RUN: not python3 %S/check_ascend_queue_lifetime.py %t/missing.mlir 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not python3 %S/check_ascend_queue_lifetime.py %t/double-free.mlir 2>&1 | FileCheck %s --check-prefix=DOUBLE

// GOOD: queue_lifetime.ok good.mlir deque=1 free=1
// MISSING: missing.mlir:{{[0-9]+}}: missing free_tensor for deque result %1 from queue %arg0
// DOUBLE: double-free.mlir:{{[0-9]+}}: duplicate free_tensor for deque result %1 from queue %arg0

//--- good.mlir
module {
  func.func @ok(%arg0: !ascendc.queue<tensor<64xf16>, 1>) {
    %1 = ascendc.deque %arg0 : (!ascendc.queue<tensor<64xf16>, 1>) -> !ascendc.local_tensor<tensor<64xf16>>
    ascendc.free_tensor %arg0, %1 : !ascendc.queue<tensor<64xf16>, 1>, !ascendc.local_tensor<tensor<64xf16>>
    return
  }
}

//--- missing.mlir
module {
  func.func @missing(%arg0: !ascendc.queue<tensor<64xf16>, 1>) {
    %1 = ascendc.deque %arg0 : (!ascendc.queue<tensor<64xf16>, 1>) -> !ascendc.local_tensor<tensor<64xf16>>
    return
  }
}

//--- double-free.mlir
module {
  func.func @double_free(%arg0: !ascendc.queue<tensor<64xf16>, 1>) {
    %1 = ascendc.deque %arg0 : (!ascendc.queue<tensor<64xf16>, 1>) -> !ascendc.local_tensor<tensor<64xf16>>
    ascendc.free_tensor %arg0, %1 : !ascendc.queue<tensor<64xf16>, 1>, !ascendc.local_tensor<tensor<64xf16>>
    ascendc.free_tensor %arg0, %1 : !ascendc.queue<tensor<64xf16>, 1>, !ascendc.local_tensor<tensor<64xf16>>
    return
  }
}
