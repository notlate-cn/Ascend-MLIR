// RUN: python3 -c "from pathlib import Path; src = Path(r'%S/../../examples/matmul-add-leakyrelu/step7_cann.mlir').read_text(); src = src.replace('1.000000e-03 : f32', '0.000000e+00 : f32', 1); Path(r'%t').write_text(src)"
// RUN: afir-translate -mlir-to-cann %t | FileCheck %s

// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: mm.SetBias(biasGM);
// CHECK: Relu(outLocal, inLocal, count);
// CHECK-NOT: LeakyRelu(

// Intentionally empty: this test mutates the checked-in mix example to verify
// that translator-side epilogue inference switches from LeakyRelu(alpha) to
// Relu when the duplicated scalar constant becomes zero.
