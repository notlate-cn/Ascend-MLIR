// RUN: not afir-translate -mlir-to-cann %S/cann-translate-mix-unsupported-vector-input.mlir 2>&1 | FileCheck %s

// CHECK: unsupported vector op in single-chain mix emitter
