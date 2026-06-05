// RUN: ascend-mlir-opt %s --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @manual_memory_copy_kernel(%arg0: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<4x8xf32>)
      outs(%empty : tensor<4x8xf32>)
      attrs = {
        ascend.kernel = "kernel_0",
        ascend.op_role = "memory",
        ascend.primary = true
      } {
  ^bb0(%x: f32, %out_elem: f32):
    linalg.yield %x : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}

// CHECK: SchedulePatternView:
// CHECK-NEXT: kernel = kernel_0
// CHECK-NEXT: ops = 1
// CHECK-NEXT: primary_ops = 1
// CHECK-NEXT: dominant_role = memory
// CHECK: TemplateRegistry:
// CHECK-NEXT: kernel = kernel_0
// CHECK-NEXT: matches = 1
// CHECK-NEXT: template = memory_copy/single_tile_per_block
// CHECK: ScheduleSearch:
// CHECK-NEXT: kernel = kernel_0
// CHECK-NEXT: generated = 1
// CHECK-NEXT: kept = 1
// CHECK: Schedule report
// CHECK: op_role = "memory"
// CHECK: schedule_family = "memory_copy"
// CHECK: linalg.generic
// CHECK-SAME: ascend.schedule.family = "memory_copy"
