// REQUIRES: ascend_env
// XFAIL: *
// Current blocker: full transformer runtime-session reaches CANN artifact
// compilation, but generated scalar fallback still emits unsupported AscendC
// forms such as GlobalTensor->GlobalTensor DataCopy, scalar Exp, double scalar
// casts, and residual GM pointer views.
// RUN: bash %S/../../../examples/transformer/run-mainline.sh --runtime-e2e --log | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.full_codegen=pass
// CHECK: transformer_dynamic.runtime_session=pass
// CHECK: session.result=success
// CHECK: session.validation=pass
// CHECK: transformer_dynamic.validation=pass
