// RUN: afir-opt --auto-fuse-group-analysis %s | FileCheck %s

// Pre-pass step in GroupAnalysisPass rematerializes multi-use tensor.empty
// ops so each consumer gets its own buffer.  Without this, torch-mlir's
// CSE-by-type sharing of tensor.empty (e.g. 12 layers' QKV-projection
// transposes all outputting the same [64x192] shape) becomes one memref
// with N writes after bufferization → only the last write wins, prior
// consumers read NaN garbage.

// CHECK-LABEL: func.func @shared_empty
// Two transposes that used to share %0 must now have their own empty ops.
// After unsharing, the IR must contain TWO tensor.empty<8x4xf32> ops.
// CHECK-COUNT-2: tensor.empty() : tensor<8x4xf32>

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @shared_empty(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>)
    -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %e = tensor.empty() : tensor<8x4xf32>
  %t1 = linalg.transpose ins(%a : tensor<4x8xf32>) outs(%e : tensor<8x4xf32>) permutation = [1, 0]
  %t2 = linalg.transpose ins(%b : tensor<4x8xf32>) outs(%e : tensor<8x4xf32>) permutation = [1, 0]
  return %t1, %t2 : tensor<8x4xf32>, tensor<8x4xf32>
}
