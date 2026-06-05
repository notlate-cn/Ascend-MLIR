// RUN: ascend-mlir-opt %s --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s

func.func @roles_array_only(%arg0: tensor<64xf16>,
                            %arg1: tensor<64xf16>) -> tensor<64xf16>
    attributes {ascend.normalized = true} {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_roles = ["Primary", "Vector", "Injective"],
      ascend.primary = true
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK: Schedule report
// CHECK: schedule_family = "vector_generic"
// CHECK: schedule_template = "single_tile_per_block"
// CHECK: ascend.schedule.family = "vector_generic"
