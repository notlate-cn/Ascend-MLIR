// ============================================================
// STAGE 2: Transform Dialect Tiling - 使用变换方言进行分块
//
// 本阶段使用 Transform Dialect 描述分块策略：
//   输入：step0_input.mlir 的两个独立的 linalg.generic（Op1 和 Op2）
//   迭代器类型：["parallel", "parallel"]
//     * d0 = M (Parallel) - 可并行维度，用于核间分发
//     * d1 = N (Parallel) - 内层维度，不切分（整体搬运）
//
// Tiling 结构（仅对 Parallel 轴 d0 进行两级切分）：
//   - TB 层：核间并行，每核负责 TB_M 行（标记 ascendc.parallel）
//   - Tb 层：UB（Unified Buffer）批次，每批 Tb_M 行（搬运粒度）
//   - d1（N）轴：不切分，整 N 在 UB 内处理
//
// Op1 和 Op2 通过 library_call 属性区分，各自独立 tile：
//   - Op1: library_call = "broadcast_add"
//   - Op2: library_call = "broadcast_mul"
//
// Concat 语义通过 tensor.insert_slice 的 offset 隐式携带：
//   Op1 结果写入 output[0:M, :]（offset=0）
//   Op2 结果写入 output[M:2M, :]（offset=M）
//
// 函数签名扩展：追加两个 index 参数 TB_M, Tb_M
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

// 广播映射：只访问 d0 轴（用于 1D 输入）
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射：访问 d0, d1（用于 2D 输入和输出）
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // ==========================================================
  // 主计算函数：广播逐元素运算 + Concat
  // ==========================================================
  func.func @ewop_broadcast_concat(
      %input_a : tensor<?xf16>,
      %input_b : tensor<?x?xf16>,
      %input_c : tensor<?xf16>,
      %input_d : tensor<?x?xf16>
  ) -> tensor<?x?xf16> {

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index
    %c2    = arith.constant 2 : index

    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>
    %dim_n = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

    %dim_2m = arith.muli %dim_m, %c2 : index
    %empty_out = tensor.empty(%dim_2m, %dim_n) : tensor<?x?xf16>

    %slice_c = tensor.extract_slice %empty_out[0, 0][%dim_m, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %tensor_c = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "broadcast_add"
    } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%slice_c : tensor<?x?xf16>) {
    ^bb0(%a_val: f16, %b_val: f16, %c_out: f16):
      %sum = arith.addf %a_val, %b_val : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>
    %out_with_c = tensor.insert_slice %tensor_c into %empty_out
        [0, 0] [%dim_m, %dim_n] [1, 1]
        : tensor<?x?xf16> into tensor<?x?xf16>

    %slice_d = tensor.extract_slice %out_with_c[%dim_m, 0][%dim_m, %dim_n][1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %tensor_d = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #full_access_map],
      iterator_types = ["parallel", "parallel"],
      library_call = "broadcast_mul"
    } ins(%input_c, %input_d : tensor<?xf16>, tensor<?x?xf16>)
      outs(%slice_d : tensor<?x?xf16>) {
    ^bb0(%c_val: f16, %d_val: f16, %e_out: f16):
      %prod = arith.mulf %c_val, %d_val : f16
      linalg.yield %prod : f16
    } -> tensor<?x?xf16>
    %out_final = tensor.insert_slice %tensor_d into %out_with_c
        [%dim_m, 0] [%dim_m, %dim_n] [1, 1]
        : tensor<?x?xf16> into tensor<?x?xf16>

    return %out_final : tensor<?x?xf16>
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

    // ---- Step 2: 分别匹配 Op1 和 Op2（通过 library_call 属性区分）----
    %op1 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "broadcast_add"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %op2 = transform.structured.match ops{["linalg.generic"]}
        attributes{library_call = "broadcast_mul"} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // ---- 公共标注参数 ----
    %true_param = transform.param.constant true -> !transform.any_param
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant
        "AiCore.Vector" -> !transform.any_param

    // ══════════════ Op1（broadcast+add）Tiling ══════════════

    // TB 层切分（d0=M，tile_size=0 表示 d1=N 不切）
    %op1_tb, %loop_op1_tb =
        transform.structured.tile_using_for %op1
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_op1_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    // Tb 层切分（UB 批次粒度）
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

    // ══════════════ Op2（broadcast+mul）Tiling ══════════════

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

    transform.yield
  }
}
