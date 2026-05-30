// RUN: afir-opt --auto-fuse-group-analysis %s | FileCheck %s

// linalg.transpose ins(arith.constant) → folded into a new arith.constant
// at compile time. Eliminates one runtime dispatch per matched weight in
// transformer-style networks (GPT-2 small: 49/97 transposes, ~50%).

// CHECK-LABEL: func.func @weight_transpose
// The original arith.constant<2x3> + linalg.transpose perm=[1,0] should be
// gone; replaced by a single arith.constant<3x2> with the permuted values.
// CHECK-NOT: linalg.transpose
// CHECK: arith.constant
// Verify the permutation actually happened: input row-major
//   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]   (shape 2x3)
// transposed to
//   [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]  (shape 3x2)
// CHECK-SAME: dense<{{\[\[1.000000e\+00, 4.000000e\+00\], \[2.000000e\+00, 5.000000e\+00\], \[3.000000e\+00, 6.000000e\+00\]\]}}> : tensor<3x2xf32>

func.func @weight_transpose() -> tensor<3x2xf32> {
  %w = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %init = tensor.empty() : tensor<3x2xf32>
  %t = linalg.transpose ins(%w : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) permutation = [1, 0]
  return %t : tensor<3x2xf32>
}

// -----

// Non-constant transpose (input is a func arg, not a constant) must NOT
// be folded.

// CHECK-LABEL: func.func @runtime_transpose
// CHECK: linalg.transpose

func.func @runtime_transpose(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %init = tensor.empty() : tensor<3x2xf32>
  %t = linalg.transpose ins(%x : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) permutation = [1, 0]
  return %t : tensor<3x2xf32>
}
