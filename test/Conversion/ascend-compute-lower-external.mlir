// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

// CHECK: func.func private @external_kernel_input
func.func private @external_kernel_input(memref<?xf32>) -> ()
