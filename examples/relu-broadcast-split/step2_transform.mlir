// ============================================================
// STAGE 2: Transform Dialect Tiling - 使用变换方言进行分块
//
// 本阶段使用 Transform Dialect 描述分块策略：
//   Op1 (relu+broadcast_add): iterator = ["parallel", "parallel"]
//     * 融合后由 linalg-fuse-elementwise-ops 得到
//     * d0=M (Parallel)，d1=N (Parallel)
//     * 仅对 d0（M 轴）做两级切分 TB/Tb，d1（N）轴不切
//
//   Op2 (split_scale0): iterator = ["parallel", "parallel"]
//     * 输入是 C[:, 0:N/2]（前半列），scale0[N/2] 列广播
//     * 仅对 d0（M 轴）做两级切分，d1（N/2）轴不切
//
//   Op3 (split_scale1): iterator = ["parallel", "parallel"]
//     * 输入是 C[:, N/2:N]（后半列），scale1[N/2] 列广播
//     * 仅对 d0（M 轴）做两级切分，d1（N/2）轴不切
//
// Tiling 结构：
//   - TB 层：核间并行，每核负责 TB_M 行（标记 ascendc.parallel）
//   - Tb 层：UB（Unified Buffer）批次，每批 Tb_M 行（搬运粒度）
//   - N/N/2 轴：不切分，整体在 UB 内处理
//
// Op2 和 Op3 通过 library_call 属性区分：
//   - Op2: library_call = "split_scale0"
//   - Op3: library_call = "split_scale1"
//
// 注意：step1 的融合会将 Op1（relu+broadcast_add）融合为单 generic，
//       但可能无法通过属性匹配，所以 Transform 直接对所有 linalg.generic 操作
//       通过 iterator 类型或 library_call 区分。
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

// 行广播映射：1D 输入只访问 d0
#row_broadcast_map = affine_map<(d0, d1) -> (d0)>
// 列广播映射：1D 输入只访问 d1
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// 完整访问映射
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // ==========================================================
  // 主计算函数（融合后版本，step1_fused.mlir 的内容）
  // ==========================================================
  func.func @ewop_broadcast_split(
      %input_a  : tensor<?x?xf16>,
      %bias     : tensor<?xf16>,
      %scale0   : tensor<?xf16>,
      %scale1   : tensor<?xf16>
  ) -> (tensor<?x?xf16>, tensor<?x?xf16>) {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index

    %dim_m  = tensor.dim %input_a, %idx_0 : tensor<?x?xf16>
    %dim_n  = tensor.dim %input_a, %idx_1 : tensor<?x?xf16>
    %dim_hn = tensor.dim %scale0, %idx_0  : tensor<?xf16>

    // Op1: relu(A) + broadcast_row(bias) → C
    %zero_f16 = arith.constant 0.0 : f16
    %empty_c = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#full_access_map, #row_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "relu_broadcast_add"
    } ins(%input_a, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_c : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %b_val: f16, %c_out: f16):
      %relu_a = arith.maximumf %a_val, %zero_f16 : f16
      %result  = arith.addf %relu_a, %b_val : f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    // Op2: C[:, 0:N/2] * broadcast_col(scale0) → out0
    %c_slice0 = tensor.extract_slice %tensor_c[0, 0][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %empty_out0 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out0 = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "split_scale0"
    } ins(%c_slice0, %scale0 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out0 : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %s_val: f16, %o_out: f16):
      %prod = arith.mulf %c_val, %s_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    // Op3: C[:, N/2:N] * broadcast_col(scale1) → out1
    %c_slice1 = tensor.extract_slice %tensor_c[0, %dim_hn][%dim_m, %dim_hn][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %empty_out1 = tensor.empty(%dim_m, %dim_hn) : tensor<?x?xf16>
    %out1 = linalg.generic {
      indexing_maps = [#full_access_map, #col_broadcast_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "split_scale1"
    } ins(%c_slice1, %scale1 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out1 : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %s_val: f16, %o_out: f16):
      %prod = arith.mulf %c_val, %s_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>

    return %out0, %out1 : tensor<?x?xf16>, tensor<?x?xf16>
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
        attributes{library_call = "relu_broadcast_add"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %op2 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "split_scale0"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %op3 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "split_scale1"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // ---- 公共标注参数 ----
    %true_param = transform.param.constant true -> !transform.any_param
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant
        "AiCore.Vector" -> !transform.any_param

    // ══════════════ Op2（split_scale0）Tiling ══════════════

    // TB 层切分（d0=M，tile_size=0 表示 d1=N/2 不切）
    %op2_tb, %loop_op2_tb =
        transform.structured.tile_using_for %op2
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op2_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    // Tb 层切分（UB 批次粒度）
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

    // ══════════════ Op3（split_scale1）Tiling ══════════════

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

    // ══════════════ Op1（relu+broadcast_add）Tiling ══════════════

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

    transform.yield
  }
}
