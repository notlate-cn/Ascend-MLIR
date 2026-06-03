// RUN: afir-opt %s --ascend-realize='materialization-mode=memory-space-annotate' | FileCheck %s

#col = affine_map<(d0, d1) -> (d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @gather_output_bridge
// CHECK: %[[VEC:.*]] = memref.alloc() {{.*}} : memref<4x3xf16, 10 : i32>
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[VEC]] : memref<4x3xf16, 10 : i32>)
// CHECK-SAME: ascend.schedule.tail_plan
// CHECK-SAME: buffering = "separate_tail_buffer"
// CHECK-SAME: selected = "masked_tail"
// CHECK: memref.copy %[[VEC]], %{{.*}} : memref<4x3xf16, 10 : i32> to memref<4x3xf16>
func.func @gather_output_bridge(%data: memref<4x8xf16>,
                                %indices: memref<3xi64>,
                                %bias: memref<3xf16>) -> memref<4x3xf16>
    attributes {ascend.normalized = true} {
  %cst = arith.constant 0.0 : f16
  %out = memref.alloc() {alignment = 64 : i64} : memref<4x3xf16>
  linalg.generic {
      indexing_maps = [#col, #col, #id],
      iterator_types = ["parallel", "parallel"]}
      ins(%indices, %bias : memref<3xi64>, memref<3xf16>)
      outs(%out : memref<4x3xf16>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.schedule_contract = "generic_tiled_loop",
               ascend.schedule.tail_plan = [
                 {axis = 0 : i64,
                  selected = "masked_tail",
                  affected = ["data_copy", "vector_compute", "write_back", "gather_index"],
                  align = 16 : i64,
                  buffering = "separate_tail_buffer"}],
               ascend.schedule.tail_policies = ["masked_tail", "full_extent"],
               gather_dim = 1 : i64} {
  ^bb0(%idx: i64, %b: f16, %o: f16):
    %i = linalg.index 0 : index
    %idx_cast = arith.index_cast %idx : i64 to index
    %loaded = memref.load %data[%i, %idx_cast] : memref<4x8xf16>
    %relu = arith.maximumf %loaded, %cst : f16
    %sum = arith.addf %relu, %b : f16
    linalg.yield %sum : f16
  }
  return %out : memref<4x3xf16>
}
