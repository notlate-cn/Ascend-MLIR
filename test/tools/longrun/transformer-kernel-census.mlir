// REQUIRES: ascend_env
// REQUIRES: ascend_longrun
// Full transformer kernel census. This is a longrun diagnostic gate that
// regenerates the 1x1 runtime E2E artifacts, then summarizes the compiler
// manifest, run manifest, DAG shape, and runtime profile markers.
// RUN: TRANSFORMER_CENSUS_BATCH=1 TRANSFORMER_CENSUS_SEQ=1 RUN_TIMEOUT=600s TRANSFORMER_CENSUS_TIMEOUT=900s bash %S/transformer_kernel_census.sh | FileCheck %s

// CHECK: INFO: transformer kernel census entry
// CHECK: transformer_census.batch=1
// CHECK: transformer_census.seq=1
// CHECK: transformer_census.kernel_count=53
// CHECK: transformer_census.task_count=53
// CHECK: transformer_census.graph_edges=47
// CHECK: transformer_census.kind.vec=49
// CHECK: transformer_census.kind.cube=1
// CHECK: transformer_census.kind.mix=3
// CHECK: transformer_census.root_tasks=26
// CHECK: transformer_census.leaf_tasks=16
// CHECK: transformer_census.critical_path_depth=11
// CHECK: transformer_census.runtime_input_roots=1
// CHECK: transformer_census.prepack_candidate_roots=25
// CHECK: transformer_census.profile.count=53
// CHECK: ALL TRANSFORMER KERNEL CENSUS PASSED
