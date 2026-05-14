// RUN: sed -n '1,/\/\/ -----/p' %s | afir-opt --ascend-normalize | FileCheck %s
// RUN: sed -n '/\/\/ CF-BR-BEGIN/,/\/\/ CF-BR-END/p' %s | not afir-opt --ascend-normalize 2>&1 | FileCheck %s --check-prefix=CF-BR-ERR
// RUN: sed -n '/\/\/ AFFINE-BEGIN/,/\/\/ AFFINE-END/p' %s | not afir-opt --ascend-normalize 2>&1 | FileCheck %s --check-prefix=AFFINE-ERR
// RUN: sed -n '/\/\/ UNKNOWN-BEGIN/,/\/\/ UNKNOWN-END/p' %s | not afir-opt --allow-unregistered-dialect --ascend-normalize 2>&1 | FileCheck %s --check-prefix=UNKNOWN-ERR

func.func @valid(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
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

// CHECK-LABEL: func.func @valid
// CHECK-SAME: ascend.normalized = true

func.func @valid_shape_guard(%arg0: index) {
  %c0 = arith.constant 0 : index
  %ok = arith.cmpi sge, %arg0, %c0 : index
  cf.assert %ok, "nonnegative dynamic dimension"
  return
}

// CHECK-LABEL: func.func @valid_shape_guard
// CHECK-SAME: ascend.normalized = true

// -----

// CF-BR-BEGIN
func.func @cf_branch() {
  cf.br ^bb1
^bb1:
  return
}
// CF-BR-END

// CF-BR-ERR: unsupported dialect before Kernelize

// -----

// AFFINE-BEGIN
func.func @affine_op(%i: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
  return %0 : index
}
// AFFINE-END

// AFFINE-ERR: unsupported dialect before Kernelize

// -----

// UNKNOWN-BEGIN
"test.unknown"() : () -> ()
// UNKNOWN-END

// UNKNOWN-ERR: unsupported dialect before Kernelize
