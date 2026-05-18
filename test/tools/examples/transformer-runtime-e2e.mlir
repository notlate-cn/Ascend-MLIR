// REQUIRES: ascend_env
// XFAIL: *
// Current blocker: full transformer runtime-session reaches CANN artifact
// compilation and enters sim, but runtime validation is still too slow / not
// closed for the examples gate.  Keep this bounded while it remains XFAIL.
// RUN: timeout 30s bash %S/../../../examples/transformer/run-mainline.sh --runtime-e2e --log | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.full_codegen=pass
// CHECK: transformer_dynamic.runtime_session=pass
// CHECK: session.result=success
// CHECK: session.validation=pass
// CHECK: transformer_dynamic.validation=pass
