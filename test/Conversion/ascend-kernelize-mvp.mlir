// RUN: ascend-mlir-opt --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' %s 2>&1 | FileCheck %s
// RUN: sed -n '/\/\/ MISSING-NORMALIZE-BEGIN/,/\/\/ MISSING-NORMALIZE-END/p' %s | not ascend-mlir-opt --ascend-kernelize 2>&1 | FileCheck %s --check-prefix=MISSING-NORMALIZE

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

// CHECK: Kernelize report
// CHECK: op_role = "vector"
// CHECK: kernel_pattern = "kernel_0"
// CHECK: primary_ops = 1
// CHECK: ascend.kernel

// -----

// MISSING-NORMALIZE-BEGIN
func.func @missing_normalize() {
  return
}
// MISSING-NORMALIZE-END

// MISSING-NORMALIZE: requires ascend.normalized
