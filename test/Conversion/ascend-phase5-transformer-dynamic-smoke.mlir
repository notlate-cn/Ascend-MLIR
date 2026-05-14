// RUN: bash %S/../../examples/transformer/run-mainline.sh | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.transpose_kernelize_generalization=pass
// CHECK: transformer_dynamic.multi_kernel_func_metadata=per_kernel
// CHECK: transformer_dynamic.full_codegen=pass
