// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK-LABEL: extern "C" __global__ __aicore__ void scalar_math(
// CHECK-NOT: double
// CHECK: constexpr float
// CHECK: afir_scalar_exp(
// CHECK-NOT: AscendC::Exp
// CHECK: afir_scalar_rsqrt(
// CHECK-NOT: AscendC::Rsqrt

module {
  func.func @scalar_math(
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["N"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    %c0 = arith.constant 0 : index
    %eps64 = arith.constant 1.000000e-05 : f64
    %eps = arith.truncf %eps64 : f64 to f32
    %x = memref.load %input[%c0] : memref<?xf32>
    %e = math.exp %x : f32
    %r = math.rsqrt %e : f32
    %y = arith.addf %r, %eps : f32
    memref.store %y, %output[%c0] : memref<?xf32>
    func.return
  }
}
