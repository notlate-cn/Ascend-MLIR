// RUN: rm -f %t.json && afir-opt %s --emit-network-json=path=%t.json
// RUN: FileCheck %s < %t.json

module {
  func.func private @kernel_group0(tensor<8xf16>) -> tensor<8xf16>
  func.func private @__aclnn_softmax(tensor<?xf16>) -> tensor<?xf16>
      attributes {aclnn.op = "Softmax"}
  func.func private @kernel_group1(tensor<8xf16>) -> tensor<8xf16>

  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    %a = call @kernel_group0(%x) : (tensor<8xf16>) -> tensor<8xf16>
    %ac = tensor.cast %a : tensor<8xf16> to tensor<?xf16>
    %s  = call @__aclnn_softmax(%ac) : (tensor<?xf16>) -> tensor<?xf16>
    %sc = tensor.cast %s : tensor<?xf16> to tensor<8xf16>
    %y  = call @kernel_group1(%sc) : (tensor<8xf16>) -> tensor<8xf16>
    return %y : tensor<8xf16>
  }
}

// CHECK:       "function": "model"
// CHECK:       "kernel_group0"
// CHECK:       "ascendc"
// CHECK:       "__aclnn_softmax"
// CHECK:       "aclnn"
// CHECK:       "op": "Softmax"
// CHECK:       "from": "kernel"
// CHECK:       "kernel": "__aclnn_softmax"
// CHECK:       "kernel_group1"
