// ============================================================
// STAGE 2: Transform Dialect Tiling - 使用变换方言进行分块
//
// 输入：step0_input.mlir（两个 linalg.generic + tensor.concat）
//
// Transform 序列步骤：
//   Step 0: decompose_concat → tensor.concat 展开为 empty + insert_slice
//   Step 1: 追加两个 index 参数（TB_M, Tb_M）
//   Step 2: 对所有 linalg.generic 做两级 tiling
//
// Tiling 结构：
//   - d0=M：两级切分，TB 层（核间并行）+ Tb 层（UB 批次）
//   - d1=N：不切分，整 N 在 UB 内处理
//
// 关键：decompose_concat 展开后，Op1/Op2 的结果通过 insert_slice 写入
// output[2M,N] 的对应行区间，tiling 后 epilogue data_copy 直接写到
// output 对应行，无 concat memcpy。
//
// 函数签名扩展：追加两个 index 参数 TB_M, Tb_M
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

// 广播映射：只访问 d0 轴
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // ==========================================================
  // 主计算函数（step0 原图：两个 generic + tensor.concat）
  // ==========================================================
  func.func @ewop_broadcast_concat(
      %input_a : tensor<?xf16>,
      %input_b : tensor<?x?xf16>,
      %input_c : tensor<?xf16>,
      %input_d : tensor<?x?xf16>
  ) -> tensor<?x?xf16> {

    %c0    = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %dim_m = tensor.dim %input_a, %c0 : tensor<?xf16>
    %dim_n = tensor.dim %input_b, %c1 : tensor<?x?xf16>

    // Op1: input_a[M] + input_b[M,N] → C[M,N]
    %init_c = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %result_c = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_c : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %b_val: f16, %c_out: f16):
      %sum = arith.addf %a_val, %b_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>

    // Op2: input_c[M] * input_d[M,N] → D[M,N]
    %init_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %result_d = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_c, %input_d : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_d : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %d_val: f16, %e_out: f16):
      %prod = arith.mulf %c_val, %d_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    // Concat: [C; D] → output[2M, N]
    // 由下方 decompose_concat pattern 展开为 empty + insert_slice
    %output = tensor.concat dim(0) %result_c, %result_d
        : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>

    return %output : tensor<?x?xf16>
  }

  // ==========================================================
  // Transform 调度脚本
  // ==========================================================
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 0: 展开 tensor.concat ----
    // decompose_concat 将 tensor.concat dim(0) (C, D) 展开为：
    //   %out = tensor.empty(%dim_2m, %dim_n)
    //   %out0 = tensor.insert_slice C into %out[0, 0][M, N][1, 1]
    //   %out1 = tensor.insert_slice D into %out0[M, 0][M, N][1, 1]
    // 此后 Op1/Op2 的 outs 变为 output 的对应 subview，
    // epilogue data_copy 直接写到 output 目标行，concat memcpy 完全消除。
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.decompose_concat
    } : !transform.any_op

    // ---- Step 1: 追加 2 个 index 参数（TB_M, Tb_M）----
    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ---- Step 2: 匹配所有 linalg.generic，统一 tiling ----
    %all_generics = transform.structured.match ops{["linalg.generic"]}
        in %func_new : (!transform.any_op) -> !transform.any_op

    // ---- 公共标注参数 ----
    %true_param = transform.param.constant true -> !transform.any_param
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant
        "AiCore.Vector" -> !transform.any_param

    // ══════════════ 对每个 generic 统一做两级 tiling ══════════════
    transform.foreach %all_generics : !transform.any_op {
    ^bb0(%op : !transform.any_op):
      // TB 层：d0=M 切 TB_M，d1=N 不切
      %op_tb, %loop_tb =
          transform.structured.tile_using_for %op
              tile_sizes [%tb_m_param, 0]
                  : (!transform.any_op, !transform.any_op)
              -> (!transform.any_op, !transform.any_op)

      transform.annotate %loop_tb "ascendc.parallel"
          = %true_param : !transform.any_op, !transform.any_param

      // Tb 层：UB 批次粒度
      %op_inner, %loop_inner =
          transform.structured.tile_using_for %op_tb
              tile_sizes [%tb_inner_m_param, 0]
                  : (!transform.any_op, !transform.any_op)
              -> (!transform.any_op, !transform.any_op)

      transform.annotate %loop_inner "ascendc.prologue"
          = %prologue_param : !transform.any_op, !transform.any_param
      transform.annotate %loop_inner "ascendc.epilogue"
          = %epilogue_param : !transform.any_op, !transform.any_param
      transform.annotate %op_inner "ascendc.unit"
          = %vector_unit_param : !transform.any_op, !transform.any_param

      transform.loop.hoist_loop_invariant_subsets %loop_inner
          : !transform.any_op
    }

    transform.yield
  }
}
