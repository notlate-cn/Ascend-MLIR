// REQUIRES: ascend_env, ascend_longrun
// Full transformer runtime-session E2E is intentionally longrun-only. The
// default xvm gate uses transformer-fragments plus the full transformer
// compile/translate/artifact smoke; full graph simulator runtime can be much
// slower than real NPU validation when f32 matmul falls back to vec kernels.
// RUN: RUN_TIMEOUT=3600s timeout 3900s bash %S/../../../examples/transformer/run-mainline.sh --runtime-e2e --batch 1 --seq 1 --log | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.full_codegen=pass
// CHECK: transformer_dynamic.artifact_compile=start
// CHECK: transformer_dynamic.artifact_compile=pass
// CHECK: transformer_dynamic.run_plan.tasks={{[0-9]+}}
// CHECK: transformer_dynamic.runtime_session=start
// CHECK: session.result=success
// CHECK: session.validation=pass
// CHECK: transformer_dynamic.runtime_session=pass
// CHECK: transformer_dynamic.validation=pass
