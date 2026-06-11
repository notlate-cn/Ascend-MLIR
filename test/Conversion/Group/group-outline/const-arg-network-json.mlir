// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt '--emit-network-json=path=%t/net.json' %s -o /dev/null
// RUN: FileCheck %s < %t/net.json

// emitNetworkJson must represent a constant kernel arg as a "const" source
// (scalar value inline / weight tensor via resource), not error on it.  The
// auto-fuse outliner rematerializes constants into AscendC kernels, but the
// aclnn path keeps weights/scalars as call args, so the emitter must handle
// them.  This exercises the emitter directly on a hand-written coordinator.

func.func private @kernel_group0(%s: f16, %x: tensor<8xf16>) -> tensor<8xf16>
    attributes {aclnn.op = "Scale"}

func.func @model(%a: tensor<8xf16>) -> tensor<8xf16> {
  %cst = arith.constant 2.0 : f16
  %0 = call @kernel_group0(%cst, %a) : (f16, tensor<8xf16>) -> tensor<8xf16>
  return %0 : tensor<8xf16>
}

// The scalar arg is a const source carrying the inline value.
// CHECK: "kernels"
// CHECK: "from": "const"
// CHECK: "value": 2
