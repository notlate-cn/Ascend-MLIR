// RUN: not afir-opt %S/../../examples/transformer/transformer_dynamic.mlir --ascend-compute-lower 2>&1 | FileCheck %s

// CHECK: unsupported
