// RUN: sed -n '/\/\/ ABI-BEGIN/,/\/\/ ABI-END/p' %s | ascend-mlir-opt --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s
// RUN: sed -n '/\/\/ PARALLELIZE-BEGIN/,/\/\/ PARALLELIZE-END/p' %s | ascend-mlir-opt --ascend-parallelize | FileCheck %s --check-prefix=PARALLELIZE
// RUN: sed -n '/\/\/ SERIAL-BEGIN/,/\/\/ SERIAL-END/p' %s | ascend-mlir-opt --ascend-parallelize | FileCheck %s --check-prefix=SERIAL

// CHECK-LABEL: func.func @broadcast_add_reducesum
// CHECK-SAME: %[[A:[a-z0-9]+]]: memref<?xf16>
// CHECK-SAME: %[[B:[a-z0-9]+]]: memref<?x?xf16>
// CHECK-SAME: %[[OUT:[a-z0-9]+]]: memref<16xf16
// CHECK-SAME: %[[WS:[a-z0-9]+]]: memref<ui8>
// CHECK-SAME: %[[TILING:[a-z0-9]+]]: !emitasc.py_struct<"TilingData"
// CHECK-SAME: cann.num_inputs = 2
// CHECK-NOT: emitasc.copy_struct
// CHECK: emitasc.member %[[TILING]] "TB_M"

// ABI-BEGIN
func.func @broadcast_add_reducesum(
    %input_a: memref<?xf16>,
    %input_b: memref<?x?xf16>,
    %dim_arg0_0: i64,
    %dim_arg1_1: i64
) -> memref<16xf16> attributes {ascendc.aicore, ascendc.global, ascendc.kernel_kind = "mix"} {
  %output = memref.alloc() : memref<16xf16>
  %tb_m = arith.constant 16 : i64
  %tb_n = arith.constant 4 : i64
  return %output : memref<16xf16>
}
// ABI-END

// PARALLELIZE-LABEL: func.func @parallel_dispatch
// PARALLELIZE: {{%[0-9a-zA-Z_]+}} = ascendc.get_block_idx : index
// PARALLELIZE-NOT: scf.for
// PARALLELIZE: scf.if
// PARALLELIZE: memref.store
// PARALLELIZE: return
// PARALLELIZE-BEGIN
func.func @parallel_dispatch(%out: memref<?xi32>, %n: index) attributes {ascendc.aicore} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %value = arith.constant 7 : i32
  scf.for %i = %c0 to %n step %c4 {
    memref.store %value, %out[%i] : memref<?xi32>
  } {ascendc.parallel = true}
  return
}
// PARALLELIZE-END

// SERIAL-LABEL: func.func @serial_loop_is_not_parallelized
// SERIAL-NOT: ascendc.get_block_idx
// SERIAL: scf.for
// SERIAL: memref.store
// SERIAL: return
// SERIAL-BEGIN
func.func @serial_loop_is_not_parallelized(%out: memref<?xi32>, %n: index) attributes {ascendc.aicore} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %value = arith.constant 7 : i32
  scf.for %i = %c0 to %n step %c4 {
    memref.store %value, %out[%i] : memref<?xi32>
  }
  return
}
// SERIAL-END
