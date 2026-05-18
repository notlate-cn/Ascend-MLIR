// RUN: rm -f %t.cpp && aclnn-backend --input %s --output %t.cpp \
// RUN:   --kernel-binaries %S/Inputs/artifacts \
// RUN:   --tilings %S/Inputs/tilings_default.json
// RUN: FileCheck %s < %t.cpp

module {
  func.func private @kernel_group0(tensor<8xf16>) -> tensor<8xf16>
  func.func private @kernel_group1(tensor<8xf16>) -> tensor<8xf16>
  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    %a = call @kernel_group0(%x) : (tensor<8xf16>) -> tensor<8xf16>
    %b = call @kernel_group1(%a) : (tensor<8xf16>) -> tensor<8xf16>
    return %b : tensor<8xf16>
  }
}

// First kernel launch uses the function argument as input.
// CHECK: TensorInfo [[IN0:[a-z0-9]+]][1] = {inputs[0]};
// CHECK: TensorInfo [[OUT0:[a-z0-9]+]][1] = {};
// CHECK: [[OUT0]][0].dtype = 1
// CHECK: hostLaunchAscendCKernel(
// CHECK-NEXT: "kernel_group0__v0"

// Second kernel launch uses the first kernel's output as input —
// this locks in the buffer-threading contract between the two calls.
// CHECK: TensorInfo [[IN1:[a-z0-9]+]][1] = {[[OUT0]][0]};
// CHECK: TensorInfo [[OUT1:[a-z0-9]+]][1] = {};
// CHECK: [[OUT1]][0].dtype = 1
// CHECK: hostLaunchAscendCKernel(
// CHECK-NEXT: "kernel_group1__v0"
