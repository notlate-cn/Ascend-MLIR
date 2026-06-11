// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt --auto-fuse-group-analysis '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.json

// A standalone cube (matmul) group is routed to the aclnn CPU-reference matmul
// (AscendC cube codegen isn't ready): the outliner stamps aclnn.op="Matmul", so
// network.json tags the kernel kind=aclnn and the host emits run_Matmul.

func.func @mm(%a: tensor<4x8xf16>, %b: tensor<8x4xf16>) -> tensor<4x4xf16> {
  %init = tensor.empty() : tensor<4x4xf16>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf16>, tensor<8x4xf16>)
       outs(%init : tensor<4x4xf16>) -> tensor<4x4xf16>
  return %c : tensor<4x4xf16>
}

// CHECK: "id": "kernel_group0"
// CHECK: "kind": "aclnn"
// CHECK: "op": "Matmul"
