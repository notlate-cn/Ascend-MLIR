// REQUIRES: ascend_env
// REQUIRES: ascend_longrun
// Full transformer shape matrix gate. This is intentionally heavier than the
// default 1x1 runtime E2E smoke because it verifies both seq and batch shape
// variation through 53 CANN artifacts and a 53-task runtime-session DAG. Enable
// it explicitly with AFIR_ENABLE_LONGRUN_TESTS=1.
// RUN: TRANSFORMER_SHAPE_MATRIX_CASES='1x1 1x2 1x4 1x16 2x2 2x4' RUN_TIMEOUT=1200s TRANSFORMER_SHAPE_TIMEOUT=2400s bash %S/transformer_shape_matrix.sh | FileCheck %s

// CHECK: INFO: transformer shape matrix test entry
// CHECK: INFO: executing 6 transformer runtime shapes
// CHECK: transformer_shape.case=batch1_seq1.validation=pass
// CHECK: transformer_shape.case=batch1_seq2.validation=pass
// CHECK: transformer_shape.case=batch1_seq4.validation=pass
// CHECK: transformer_shape.case=batch1_seq16.validation=pass
// CHECK: transformer_shape.case=batch2_seq2.validation=pass
// CHECK: transformer_shape.case=batch2_seq4.validation=pass
// CHECK: EXECUTED: 6 transformer runtime shapes
// CHECK: ALL TRANSFORMER SHAPE MATRIX PASSED
