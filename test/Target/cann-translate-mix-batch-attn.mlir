// RUN: afir-translate -mlir-to-cann %s | FileCheck %s

// CHECK: #define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__
// CHECK: #define ASCENDC_CUBE_ONLY
// CHECK-LABEL: extern "C" __global__ __aicore__ void attn_score(
// CHECK: GM_ADDR q, GM_ADDR key, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace,
// CHECK: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
// CHECK: Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
// CHECK: for (uint32_t b = 0; b < batch; ++b)
// CHECK: bmm.template IterateAll(outGM);
// CHECK: CrossCoreSetFlag<0x2, PIPE_FIX>(3);
// CHECK: CrossCoreWaitFlag(3);
// CHECK: AscendC::Add(outLocal, scoreLocal, biasLocal, count);

module {
  func.func @attn_score(
      %q: memref<?x?x?xf16>,
      %key: memref<?x?x?xf16>,
      %bias: memref<?x?x?xf32>,
      %out: memref<?x?x?xf32>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>)
      attributes {
        abi_matmul_batch_shape = [2],
        abi_matmul_epilogue_kind = "None",
        abi_matmul_has_bias = false,
        abi_matmul_layout_a = "ND",
        abi_matmul_layout_b = "ND",
        abi_matmul_layout_c = "ND",
        abi_matmul_op_kind = "batch_matmul",
        abi_matmul_trans_a = false,
        abi_matmul_trans_b = false,
        ascendc.aicore,
        ascendc.global,
        ascendc.kernel_kind = "mix",
        cann.num_inputs = 3 : i32
      } {
    return
  }
}
