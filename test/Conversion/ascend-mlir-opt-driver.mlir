// RUN: ascend-mlir-opt --help 2>&1 | FileCheck %s --check-prefix=HELP --implicit-check-not=convert-afir-to-ascir
// RUN: ascend-mlir-opt %s --ascend-normalize --ascend-kernelize | FileCheck %s

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

// HELP: --ascend-normalize
// CHECK-LABEL: func.func @elementwise
// CHECK: ascend.normalized = true
// CHECK: ascend.kernel
