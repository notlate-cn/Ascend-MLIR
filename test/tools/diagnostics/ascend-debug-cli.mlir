// RUN: bash %S/test_ascend_debug_cli.sh %s | FileCheck %s

func.func @elementwise(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

// CHECK: ascend_debug.help=ok
// CHECK: ascend_debug.collect=ok
// CHECK: ascend_debug.collect_deep=ok
// CHECK: ascend_debug.collect_graph=ok
// CHECK: ascend_debug.kernel_dag_internal=ok
// CHECK: ascend_debug.open_kernel=ok
// CHECK: ascend_debug.open_graph=ok
// CHECK: ascend_debug.locate=ok
// CHECK: ascend_debug.diff_pass=ok
// CHECK: ascend_debug.diff_fail=ok
// CHECK: ascend_debug.open_deep=ok
// CHECK: ascend_debug.stage_graph=ok
// CHECK: ascend_debug.open=ok
// CHECK: ascend_debug.manifest.stage_count=5
// CHECK: ascend_debug.stage.0=000-source.mlir
// CHECK: ascend_debug.stage.4=029-kernelize-out.mlir
// CHECK: ascend_debug.deep.stage_count=9
// CHECK: ascend_debug.deep.command_count=4
// CHECK: ascend_debug.graph.command_count=5
// CHECK: ascend_debug.graph.artifact_count=5
// CHECK: ALL ASCEND DEBUG CLI TESTS PASSED
