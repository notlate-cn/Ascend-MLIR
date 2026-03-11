// ============================================================
// STAGE 2: Transform Dialect Tiling - 使用变换方言进行分块
//
// 本阶段使用 Transform Dialect 描述分块策略：
//   Op1 (relu_bias_add): iterator = ["parallel", "parallel"]
//     * d0=M (Parallel)，d1=N (Parallel)
//     * 仅对 d0（M 轴）做两级切分 TB/Tb，d1（N）不切
//
//   Op2 (transpose): iterator = ["parallel", "parallel"]
//     * 转置 generic，迭代空间 [N,M]（d0=N, d1=M）
//     * 对 d0（N 轴）做两级切分 TB/Tb，d1（M）不切
//     * 注意：转置输入是 B[M,N]，输出是 C[N,M]
//
//   Op3 (scale_mul): iterator = ["parallel", "parallel"]
//     * d0=N (Parallel)，d1=M (Parallel)
//     * 对 d0（N 轴）做两级切分 TB/Tb，d1（M）不切
//
// Tiling 结构（对各 Op 的第一个轴做两级切分）：
//   - TB 层：核间并行，每核负责 TB_M 行（标记 ascendc.parallel）
//   - Tb 层：UB（Unified Buffer）批次，每批 Tb_M 行（搬运粒度）
//   - 第二轴：不切分，整体在 UB 内处理
//
// Op 通过 library_call 属性区分：
//   - Op1: library_call = "relu_bias_add"
//   - Op2: library_call = "transpose"
//   - Op3: library_call = "scale_mul"
//
// 函数签名扩展：追加两个 index 参数 TB_M, Tb_M
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

// 列广播映射：1D 输入只访问 d1
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>
// 转置映射：读 (d1,d0)
#transpose_map = affine_map<(d0, d1) -> (d1, d0)>

module attributes {transform.with_named_sequence} {

  // ==========================================================
  // 主计算函数（step1 融合后版本）
  // ==========================================================
  func.func @ewop_broadcast_transpose(
      %input_a : tensor<?x?xf16>,
      %bias    : tensor<?xf16>,
      %scale   : tensor<?xf16>
  ) -> tensor<?x?xf16> {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?x?xf16>
    %dim_n = tensor.dim %input_a, %idx_1 : tensor<?x?xf16>

    // Op1: relu(A) + broadcast_col(bias) → B[M,N]
    %zero_f16 = arith.constant 0.0 : f16
    %empty_b = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %tensor_b = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "relu_bias_add"
    } ins(%input_a, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_b : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %bias_val: f16, %b_out: f16):
      %relu_a = arith.maximumf %a_val, %zero_f16 : f16
      %result  = arith.addf %relu_a, %bias_val : f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    // Op2: Transpose B[M,N] → C[N,M]
    %empty_c = tensor.empty(%dim_n, %dim_m) : tensor<?x?xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#transpose_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "transpose"
    } ins(%tensor_b : tensor<?x?xf16>)
      outs(%empty_c : tensor<?x?xf16>) {
    ^bb0(%b_val: f16, %c_out: f16):
      linalg.yield %b_val : f16
    } -> tensor<?x?xf16>

    // Op3: C[N,M] * broadcast_col(scale[M]) → D[N,M]
    %empty_d = tensor.empty(%dim_n, %dim_m) : tensor<?x?xf16>
    %tensor_d = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "scale_mul"
    } ins(%tensor_c, %scale : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_d : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %s_val: f16, %d_out: f16):
      %prod = arith.mulf %c_val, %s_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    return %tensor_d : tensor<?x?xf16>
  }

  // ==========================================================
  // Transform 调度脚本
  // ==========================================================
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 1: 匹配 func.func，追加 2 个 index 参数 ----
    // 参数: TB_M（核间分块大小），Tb_M（UB 批次大小）
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ---- Step 2: 通过 library_call 属性分别匹配各 Op ----
    %op1 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "relu_bias_add"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %op2 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "transpose"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %op3 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "scale_mul"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // ---- 公共标注参数 ----
    %true_param = transform.param.constant true -> !transform.any_param
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant
        "AiCore.Vector" -> !transform.any_param

    // ══════════════ Op1（relu_bias_add）Tiling ══════════════
    // 对 d0=M 轴做两级切分

    %op1_tb, %loop_op1_tb =
        transform.structured.tile_using_for %op1
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op1_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %op1_inner, %loop_op1_inner =
        transform.structured.tile_using_for %op1_tb
            tile_sizes [%tb_inner_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op1_inner "ascendc.prologue"
        = %prologue_param : !transform.any_op, !transform.any_param
    transform.annotate %loop_op1_inner "ascendc.epilogue"
        = %epilogue_param : !transform.any_op, !transform.any_param
    transform.annotate %op1_inner "ascendc.unit"
        = %vector_unit_param : !transform.any_op, !transform.any_param

    transform.loop.hoist_loop_invariant_subsets %loop_op1_inner
        : !transform.any_op

    // ══════════════ Op2（transpose）Tiling ══════════════
    // 对 d0=N 轴做两级切分（输出 C[N,M] 的行）

    %op2_tb, %loop_op2_tb =
        transform.structured.tile_using_for %op2
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op2_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %op2_inner, %loop_op2_inner =
        transform.structured.tile_using_for %op2_tb
            tile_sizes [%tb_inner_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op2_inner "ascendc.prologue"
        = %prologue_param : !transform.any_op, !transform.any_param
    transform.annotate %loop_op2_inner "ascendc.epilogue"
        = %epilogue_param : !transform.any_op, !transform.any_param
    transform.annotate %op2_inner "ascendc.unit"
        = %vector_unit_param : !transform.any_op, !transform.any_param

    transform.loop.hoist_loop_invariant_subsets %loop_op2_inner
        : !transform.any_op

    // ══════════════ Op3（scale_mul）Tiling ══════════════
    // 对 d0=N 轴做两级切分

    %op3_tb, %loop_op3_tb =
        transform.structured.tile_using_for %op3
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op3_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %op3_inner, %loop_op3_inner =
        transform.structured.tile_using_for %op3_tb
            tile_sizes [%tb_inner_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op3_inner "ascendc.prologue"
        = %prologue_param : !transform.any_op, !transform.any_param
    transform.annotate %loop_op3_inner "ascendc.epilogue"
        = %epilogue_param : !transform.any_op, !transform.any_param
    transform.annotate %op3_inner "ascendc.unit"
        = %vector_unit_param : !transform.any_op, !transform.any_param

    transform.loop.hoist_loop_invariant_subsets %loop_op3_inner
        : !transform.any_op

    transform.yield
  }
}
