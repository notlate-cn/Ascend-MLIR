// ============================================================
// transform_tile_and_fuse_3level.mlir — 三级 Tiling 与融合调度
//
// 本文件定义了 matmul-add-relu 的三级 Tiling 和算子融合策略
//
// 计算图: output = ReLU(matmul(A, B) + bias)
//
// Tiling 层级：
//   - TB (Tile Block): 核间分块，映射到多核并行
//   - Tb (Tile block inner): 核内分块，UB 缓冲区粒度
//   - t_K (K tile): K 维度分块，Cube 计算粒度
//
// 融合策略：
//   - matmul + add + relu 融合为单个计算单元
//   - 消除中间张量，减少内存访问
//
// 内存层级映射：
//   - GM → A1/B1 (L1) → A2/B2 (L0) → CO1 (L0C)
//   - CO1 → VECIN → VECOUT → GM
// ============================================================

module attributes { transform.with_named_sequence } {

  // ----------------------------------------------------------
  // @tile_and_fuse_3level: 三级 Tiling 与融合
  // ----------------------------------------------------------
  transform.named_sequence @tile_and_fuse_3level(
      %root: !transform.any_op { transform.readonly }
  ) {
    // ---- Step 1: 匹配函数 ----
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // ---- Step 2: 添加 Tile 参数 ----
    // 添加 5 个 tile 参数: TB_M, TB_N, Tb_M, Tb_N, t_K
    %func_with_tiles, %tb_m_param, %tb_n_param, %tb_m_inner_param, %tb_n_inner_param, %t_k_param =
        transform.func.add_index_args %func, 5
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op,
                !transform.any_op, !transform.any_op, !transform.any_op)

    // ---- Step 3: 匹配 linalg.matmul ----
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_with_tiles
        : (!transform.any_op) -> !transform.any_op

    // ---- Step 4: TB 层 Tiling (核间分块) ----
    %matmul_tb, %loop_tb_m, %loop_tb_n =
        transform.structured.tile_using_for %matmul [%tb_m_param, %tb_n_param]
            : (!transform.any_op, !transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 标记并行性
    %true = transform.param.constant true : i1
    transform.annotate %loop_tb_m "ascendc.parallel" = %true
        : !transform.any_op, !transform.param<i1>
    transform.annotate %loop_tb_n "ascendc.parallel" = %true
        : !transform.any_op, !transform.param<i1>

    // 标记 prologue: GM -> A1/B1
    %prologue_tb = transform.param.constant "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN" : !transform.any_param
    transform.annotate %loop_tb_m "ascendc.prologue" = %prologue_tb
        : !transform.any_op, !transform.any_param

    // ---- Step 5: Tb 层 Tiling (核内分块) ----
    %matmul_tb_inner, %loop_tb_m_inner, %loop_tb_n_inner =
        transform.structured.tile_using_for %matmul_tb [%tb_m_inner_param, %tb_n_inner_param]
            : (!transform.any_op, !transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ---- Step 6: t_K 层 Tiling (K 维度分块) ----
    %matmul_t, %loop_t_k =
        transform.structured.tile_using_for %matmul_tb_inner [0, 0, %t_k_param]
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // 标记执行单元: Cube
    %cube_unit = transform.param.constant "AiCore.Cube" : !transform.any_param
    transform.annotate %matmul_t "ascendc.unit" = %cube_unit
        : !transform.any_op, !transform.any_param

    // 标记数据搬运
    %prologue_k = transform.param.constant "lhs:A1->A2,rhs:B1->B2" : !transform.any_param
    %epilogue_k = transform.param.constant "acc:CO1->VECIN" : !transform.any_param
    transform.annotate %loop_t_k "ascendc.prologue" = %prologue_k
        : !transform.any_op, !transform.any_param
    transform.annotate %loop_t_k "ascendc.epilogue" = %epilogue_k
        : !transform.any_op, !transform.any_param

    // ---- Step 7: 匹配并处理 elementwise 算子 ----
    %add = transform.structured.match ops{["linalg.elementwise"]} attributes{kind = #linalg.elementwise_kind<add>} in %func_with_tiles
        : (!transform.any_op) -> !transform.any_op

    %relu = transform.structured.match ops{["linalg.elementwise"]} attributes{kind = #linalg.elementwise_kind<max_signed>} in %func_with_tiles
        : (!transform.any_op) -> !transform.any_op

    // 标记 Vector 执行单元
    %vector_unit = transform.param.constant "AiCore.Vector" : !transform.any_param
    transform.annotate %add "ascendc.unit" = %vector_unit
        : !transform.any_op, !transform.any_param
    transform.annotate %relu "ascendc.unit" = %vector_unit
        : !transform.any_op, !transform.any_param

    // 标记 epilogue: VECOUT -> GM
    %epilogue_tb = transform.param.constant "result:VECOUT->GM" : !transform.any_param
    transform.annotate %loop_tb_n "ascendc.epilogue" = %epilogue_tb
        : !transform.any_op, !transform.any_param

    // ---- Step 8: 提升循环不变量 ----
    transform.loop.hoist_loop_invariant_subsets %loop_tb_m_inner : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %loop_tb_n_inner : !transform.any_op

    transform.yield
  }

  // ----------------------------------------------------------
  // @main: 主入口
  // ----------------------------------------------------------
  transform.named_sequence @__transform_main(
      %root: !transform.any_op { transform.readonly }
  ) {
    transform.include @tile_and_fuse_3level failures(propagate) (%root)
        : (!transform.any_op) -> ()

    transform.yield
  }

} // module
