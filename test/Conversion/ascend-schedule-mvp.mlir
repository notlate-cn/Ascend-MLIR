// RUN: sed -n '/\/\/ POSITIVE-BEGIN/,/\/\/ POSITIVE-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s
// RUN: sed -n '/\/\/ MATMUL-BEGIN/,/\/\/ MATMUL-END/p' %s | afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=MATMUL
// RUN: sed -n '/\/\/ MISSING-KERNEL-BEGIN/,/\/\/ MISSING-KERNEL-END/p' %s | not afir-opt --ascend-schedule='target-tile-policy=legacy-default' 2>&1 | FileCheck %s --check-prefix=MISSING-KERNEL

// POSITIVE-BEGIN
func.func @elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}
// POSITIVE-END

// CHECK: Schedule report
// CHECK: schedule_family = "vector_generic"
// CHECK: schedule_template = "single_tile_per_block"
// CHECK: ascend.schedule.family = "vector_generic"

// -----

// MATMUL-BEGIN
func.func @matmul(%lhs: tensor<4x8xf16>, %rhs: tensor<8x16xf16>) -> tensor<4x16xf16> {
  %empty = tensor.empty() : tensor<4x16xf16>
  %out = linalg.matmul ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
                       outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %out : tensor<4x16xf16>
}
// MATMUL-END

// MATMUL: Schedule report
// MATMUL: schedule_family = "cube_static_matmul"
// MATMUL: schedule_template = "single_tile_per_block"
// MATMUL: ascend.schedule.family = "cube_static_matmul"

// -----

// MISSING-KERNEL-BEGIN
func.func @missing_kernel(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}
// MISSING-KERNEL-END

// MISSING-KERNEL: requires ascend.kernel
