// RUN: afir-opt --auto-fuse-group-analysis %s | FileCheck %s

// Two independent linalg.fill ops share ONLY a scalar constant %cst (the fill
// value).  A shared *scalar* carries no shared tensor data, so it must NOT
// trigger horizontal fusion.  Regression for the encoder outline crash: every
// linalg.fill in a network shares the zero constant, and treating that as a
// horizontal-fusion edge collapsed all fills into one scattered, cyclic group
// that crashed --auto-fuse-group-outline.

func.func @fills_share_scalar(%e1: tensor<8xf16>, %e2: tensor<8xf16>)
    -> (tensor<8xf16>, tensor<8xf16>) {
  %cst = arith.constant 0.0 : f16
  %f1 = linalg.fill ins(%cst : f16) outs(%e1 : tensor<8xf16>) -> tensor<8xf16>
  %f2 = linalg.fill ins(%cst : f16) outs(%e2 : tensor<8xf16>) -> tensor<8xf16>
  return %f1, %f2 : tensor<8xf16>, tensor<8xf16>
}

// The two fills must land in DIFFERENT groups (no horizontal fusion on a scalar).
// CHECK: linalg.fill
// CHECK-SAME: auto_fuse.group_id = 0
// CHECK: linalg.fill
// CHECK-SAME: auto_fuse.group_id = 1
