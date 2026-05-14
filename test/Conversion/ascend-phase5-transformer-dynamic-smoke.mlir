// RUN: bash %S/../../examples/transformer/run-mainline.sh | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.transpose_kernelize_generalization=pass
// CHECK: transformer_dynamic.multi_kernel_func_metadata=fail_closed
// CHECK: transformer_dynamic.full_codegen=deferred
// CHECK: transformer_dynamic.next_gap=multi_kernel_func_schedule_metadata
