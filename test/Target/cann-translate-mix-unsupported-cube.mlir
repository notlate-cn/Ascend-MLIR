// RUN: not ascend-mlir-translate -mlir-to-cann %S/cann-translate-mix-unsupported-cube-input.mlir 2>&1 | FileCheck %s

// CHECK: unsupported cube op in single-chain mix emitter
