// RUN: rm -f %t.cpp && aclnn-backend --input %s --output %t.cpp
// RUN: FileCheck %s < %t.cpp

module {
  func.func private @__aclnn_softmax(tensor<?xf16>) -> tensor<?xf16>
      attributes {aclnn.op = "Softmax"}
  func.func @model(%x: tensor<?xf16>) -> tensor<?xf16> {
    %s = call @__aclnn_softmax(%x) : (tensor<?xf16>) -> tensor<?xf16>
    return %s : tensor<?xf16>
  }
}

// The generated entry must (a) try aclInit, (b) on failure call setHostMode(true)
// and continue, NOT exit.
// CHECK: extern "C" void network(
// CHECK: aclInit(nullptr)
// CHECK: setHostMode(true)
// CHECK-NOT: exit(
// CHECK: network_impl(
