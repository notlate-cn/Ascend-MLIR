// RUN: sed -n '/\/\/ R1-BEGIN/,/\/\/ R1-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R1
// RUN: sed -n '/\/\/ R2-BEGIN/,/\/\/ R2-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R2
// RUN: sed -n '/\/\/ R3-BEGIN/,/\/\/ R3-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R3
// RUN: sed -n '/\/\/ R4-BEGIN/,/\/\/ R4-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R4
// RUN: sed -n '/\/\/ R4-NEG-BEGIN/,/\/\/ R4-NEG-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R4-NEG --implicit-check-not='sym_name = "arg0_dim0"'
// RUN: sed -n '/\/\/ R5-BEGIN/,/\/\/ R5-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R5
// RUN: sed -n '/\/\/ R6-BEGIN/,/\/\/ R6-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=R6
// RUN: sed -n '/\/\/ STATIC-BEGIN/,/\/\/ STATIC-END/p' %s | afir-opt --ascend-normalize | FileCheck %s --check-prefix=STATIC --implicit-check-not='sym_name = "arg0_dim0"'
// RUN: sed -n '/\/\/ BAD-BEGIN/,/\/\/ BAD-END/p' %s | not afir-opt --ascend-normalize 2>&1 | FileCheck %s --check-prefix=BAD

// R1-BEGIN
func.func @r1_generic_matmul_like(%lhs: tensor<?x?xf16>,
                                  %rhs: tensor<?x?xf16>,
                                  %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (k, n)>,
        affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out : tensor<?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %acc: f16):
      %product = arith.mulf %x, %y : f16
      %sum = arith.addf %acc, %product : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}
// R1-END

// R1-LABEL: func.func @r1_generic_matmul_like
// R1: ascend.symbol_constraints
// R1-DAG: sym_name = "arg0_dim0"
// R1-DAG: sym_name = "arg0_dim1"
// R1-DAG: sym_name = "arg1_dim1"

// R2-BEGIN
func.func @r2_named_matmul(%lhs: tensor<?x?xf16>,
                           %rhs: tensor<?x?xf16>,
                           %out: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>)
                     outs(%out : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}
// R2-END

// R2-LABEL: func.func @r2_named_matmul
// R2: ascend.symbol_constraints
// R2-DAG: sym_name = "arg0_dim0"
// R2-DAG: sym_name = "arg0_dim1"
// R2-DAG: sym_name = "arg1_dim1"

// R3-BEGIN
func.func @r3_producer_consumer(%arg0: tensor<?x?xf16>,
                                %out0: tensor<?x?xf16>,
                                %out1: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf16>)
      outs(%out0 : tensor<?x?xf16>) {
    ^bb0(%x: f16, %out: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0 : tensor<?x?xf16>)
      outs(%out1 : tensor<?x?xf16>) {
    ^bb0(%x: f16, %out: f16):
      linalg.yield %x : f16
    } -> tensor<?x?xf16>
  return %1 : tensor<?x?xf16>
}
// R3-END

// R3-LABEL: func.func @r3_producer_consumer
// R3: ascend.symbol_constraints
// R3-DAG: sym_name = "arg0_dim0"
// R3-DAG: sym_name = "arg0_dim1"

// R4-BEGIN
func.func @r4_extract_full_slice(%src: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %src, %c0 : tensor<?x?xf16>
  %c1 = arith.constant 1 : index
  %n = tensor.dim %src, %c1 : tensor<?x?xf16>
  %slice = tensor.extract_slice %src[%c0, %c0][%m, %n][1, 1]
      : tensor<?x?xf16> to tensor<?x?xf16>
  return %slice : tensor<?x?xf16>
}
// R4-END

// R4-LABEL: func.func @r4_extract_full_slice
// R4: ascend.symbol_constraints
// R4-DAG: sym_name = "arg0_dim0"
// R4-DAG: sym_name = "arg0_dim1"

// R4-NEG-BEGIN
func.func @r4_strided_slice_does_not_merge(%src: tensor<?xf16>) -> tensor<?xf16> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %src, %c0 : tensor<?xf16>
  %slice = tensor.extract_slice %src[%c0][%m][2]
      : tensor<?xf16> to tensor<?xf16>
  return %slice : tensor<?xf16>
}
// R4-NEG-END

// R4-NEG-LABEL: func.func @r4_strided_slice_does_not_merge
// R4-NEG: ascend.symbol_constraints = []

// R5-BEGIN
func.func @r5_tensor_dim_empty(%src: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %src, %c0 : tensor<?x?xf16>
  %n = tensor.dim %src, %c1 : tensor<?x?xf16>
  %empty = tensor.empty(%m, %n) : tensor<?x?xf16>
  return %empty : tensor<?x?xf16>
}
// R5-END

// R5-LABEL: func.func @r5_tensor_dim_empty
// R5: ascend.symbol_constraints
// R5-DAG: sym_name = "arg0_dim0"
// R5-DAG: sym_name = "arg0_dim1"

// R6-BEGIN
func.func @r6_linalg_broadcast(%input: tensor<?x?xf16>,
                               %out: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  %0 = linalg.broadcast ins(%input : tensor<?x?xf16>)
      outs(%out : tensor<?x?x?xf16>) dimensions = [0]
  return %0 : tensor<?x?x?xf16>
}
// R6-END

// R6-LABEL: func.func @r6_linalg_broadcast
// R6: ascend.symbol_constraints
// R6-DAG: sym_name = "arg0_dim0"
// R6-DAG: sym_name = "arg0_dim1"

// STATIC-BEGIN
func.func @static_dims_are_not_members(%arg0: tensor<4x?xf16>,
                                       %out: tensor<4x?xf16>) -> tensor<4x?xf16> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<4x?xf16>)
      outs(%out : tensor<4x?xf16>) {
    ^bb0(%x: f16, %outv: f16):
      linalg.yield %x : f16
    } -> tensor<4x?xf16>
  return %0 : tensor<4x?xf16>
}
// STATIC-END

// STATIC-LABEL: func.func @static_dims_are_not_members
// STATIC: ascend.symbol_constraints
// STATIC: sym_name = "arg0_dim1"

// BAD-BEGIN
func.func @bad_existing_symbol_attr(%arg0: tensor<?xf16>)
    attributes {
      ascend.symbol_constraints = [
        {sym_name = "arg0_dim0", members = [
          {value = 0 : i64, dim = 0 : i64},
          {value = 0 : i64, dim = 0 : i64}
        ]}
      ]
    } {
  return
}
// BAD-END

// BAD: duplicate DimRef in symbol constraints
