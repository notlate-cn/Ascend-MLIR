// RUN: rm -f %t.cpp && aclnn-backend --input %s --output %t.cpp \
// RUN:   --kernel-binaries %S/Inputs/artifacts \
// RUN:   --tilings %S/Inputs/tilings_default.json
// RUN: FileCheck %s < %t.cpp

module {
  func.func private @kernel_group0(tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
  func.func @model(%a: tensor<8xf16>, %b: tensor<8xf16>) -> tensor<8xf16> {
    %r = call @kernel_group0(%a, %b) : (tensor<8xf16>, tensor<8xf16>) -> tensor<8xf16>
    return %r : tensor<8xf16>
  }
}

// CHECK-NOT: TODO: launch
// CHECK: #include "Runtime/Execution/HostLaunchHelper.h"
// CHECK: hostLaunchAscendCKernel(
// CHECK: "kernel_group0"
// CHECK: /*kernelBinariesDir=*/
// CHECK: /*tilingsPath=*/
// CHECK: extern "C" void network_set_dump_dir(const char *dir)
// CHECK: mlir::runtime::hostLaunchSetDumpIntermediatesDir(dir)
