// RUN: afir-opt --aclnn-finalize-decl %s | FileCheck %s

// Verifies that AclnnFinalizeDeclPass:
//   1. Replaces aclnn.kind = "flash_attention" with aclnn.op + aclnn.layout
//   2. Removes aclnn.kind after finalization
//   3. Leaves unrelated private funcs untouched

// -----
// CHECK: func.func private @__aclnn_flash_attention
// CHECK-SAME: aclnn.layout = "BNSD"
// CHECK-SAME: aclnn.op = "FlashAttentionScore"
// CHECK-NOT:  aclnn.kind

// CHECK-LABEL: func.func @coordinator
// CHECK:         call @__aclnn_flash_attention

// CHECK: func.func private @unrelated_helper
// CHECK-NOT: aclnn

module {
  func.func private @__aclnn_flash_attention(
      tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>,
      tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>
  ) -> tensor<?x?x?x?xf32>
      attributes {aclnn.kind = "flash_attention"}

  func.func @coordinator(%q: tensor<?x?x?x?xf32>,
                          %k: tensor<?x?x?x?xf32>,
                          %v: tensor<?x?x?x?xf32>,
                          %m: tensor<?x?x?x?xf32>,
                          %i: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %out = func.call @__aclnn_flash_attention(%q, %k, %v, %m, %i)
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>,
           tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %out : tensor<?x?x?x?xf32>
  }

  // A private func without aclnn.kind must be left unchanged.
  func.func private @unrelated_helper(%x: tensor<4xf32>) -> tensor<4xf32>
}