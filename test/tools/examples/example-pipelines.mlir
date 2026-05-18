// REQUIRES: ascend_env
// RUN: bash %S/example_pipelines.sh | FileCheck %s

// CHECK: INFO: example pipeline test entry
// CHECK: INFO: executing 10 example pipelines
// CHECK: EXECUTED: 10 example pipelines
// CHECK: ALL EXAMPLE PIPELINES PASSED
