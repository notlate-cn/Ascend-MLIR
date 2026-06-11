// RUN: rm -rf %t && mkdir -p %t
// RUN: afir-opt '--auto-fuse-group-analysis=disable-cube-fusion=true' \
// RUN:   '--auto-fuse-group-outline=output-dir=%t' %s
// RUN: FileCheck %s < %t/network.json

// batch_matmul is a cube op; with disable-cube-fusion it stays a standalone
// group and is routed to the aclnn matmul fallback.  The trailing transpose
// becomes its own (AscendC) vector group rather than fusing into the matmul.

func.func @bmm(%a: tensor<2x4x8xf16>, %b: tensor<2x8x4xf16>) -> tensor<2x4x4xf16> {
  %init = tensor.empty() : tensor<2x4x4xf16>
  %c = linalg.batch_matmul ins(%a, %b : tensor<2x4x8xf16>, tensor<2x8x4xf16>)
       outs(%init : tensor<2x4x4xf16>) -> tensor<2x4x4xf16>
  return %c : tensor<2x4x4xf16>
}

// CHECK: "kind": "aclnn"
// CHECK: "op": "Matmul"
