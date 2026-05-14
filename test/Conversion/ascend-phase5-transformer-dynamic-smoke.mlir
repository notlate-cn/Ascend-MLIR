// RUN: bash %S/../../examples/transformer/run-mainline.sh | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.rank2_transpose_closure=pass
// CHECK: transformer_dynamic.full_codegen=deferred
// CHECK: transformer_dynamic.next_gap=rank3_transpose_semantics
