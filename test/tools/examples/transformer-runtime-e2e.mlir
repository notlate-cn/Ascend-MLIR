// REQUIRES: ascend_env
// Full transformer runtime-session E2E gate: compile 53 per-kernel artifacts,
// generate a 53-task run manifest, run simulator validation, and compare only
// named golden outputs.
// RUN: RUN_TIMEOUT=180s timeout 240s bash %S/../../../examples/transformer/run-mainline.sh --runtime-e2e --batch 1 --seq 1 --log | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.full_codegen=pass
// CHECK: transformer_dynamic.artifact_compile=start
// CHECK: transformer_dynamic.artifact_compile=pass
// CHECK: transformer_dynamic.run_plan.tasks=53
// CHECK: transformer_dynamic.runtime_session=start
// CHECK: session.result=success
// CHECK: session.validation=pass
// CHECK: transformer_dynamic.runtime_session=pass
// CHECK: transformer_dynamic.validation=pass
