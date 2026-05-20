// RUN: bash %S/test_ascend_kernel_dag_viz.sh | FileCheck %s

// CHECK: INFO: ascend kernel dag viz test entry
// CHECK: ascend_kernel_dag_viz.kernel_count=5
// CHECK: ascend_kernel_dag_viz.graph_edges=4
// CHECK: ascend_kernel_dag_viz.kind.vec=4
// CHECK: ascend_kernel_dag_viz.kind.mix=1
// CHECK: ascend_kernel_dag_viz.root_tasks=2
// CHECK: ascend_kernel_dag_viz.prepack_candidate_roots=1
// CHECK: ascend_kernel_dag_viz.critical_path_depth=4
// CHECK: ascend_kernel_dag_viz.simple_fusion_edges=1
// CHECK: ascend_kernel_dag_viz.svg.contains.kernel_2=true
// CHECK: ascend_kernel_dag_viz.svg.contains.batch_matmul=true
// CHECK: ascend_kernel_dag_viz.svg.contains.1x4x128=true
// CHECK: ALL ASCEND KERNEL DAG VIZ TESTS PASSED
