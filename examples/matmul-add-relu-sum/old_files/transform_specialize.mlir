// ============================================================
// transform_specialize.mlir — Transform Dialect 特化调度
//
// 本文件定义了针对 matmul-add-relu 计算图的特化调度策略
// 使用 Transform Dialect 描述编译优化流程
//
// 计算图: output = ReLU(matmul(A, B) + bias)
//
// 调度策略：
//   1. 匹配 linalg.matmul 算子
//   2. 应用三级 Tiling (TB/Tb/t_K)
//   3. 标记并行性和执行单元
//   4. 生成优化后的 IR
//
// 关键概念：
//   - match: 匹配特定算子
//   - tile_using_for: 使用 for 循环进行分块
//   - annotate: 添加属性注解
//   - map_to_blocks: 映射到多核
// ============================================================

module attributes { transform.with_named_sequence } {

  // ----------------------------------------------------------
  // @specialize_fc_relu: 全连接+ReLU特化调度
  // ----------------------------------------------------------
  transform.named_sequence @specialize_fc_relu(
      %root: !transform.any_op { transform.readonly }
  ) {
    // ---- Step 1: 匹配 func.func ----
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // ---- Step 2: 匹配 linalg.matmul ----
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
        : (!transform.any_op) -> !transform.any_op

    // ---- Step 3: 匹配 linalg.elementwise (add) ----
    %add = transform.structured.match ops{["linalg.elementwise"]} attributes{kind = #linalg.elementwise_kind<add>} in %func
        : (!transform.any_op) -> !transform.any_op

    // ---- Step 4: 匹配 linalg.elementwise (max/ReLU) ----
    %relu = transform.structured.match ops{["linalg.elementwise"]} attributes{kind = #linalg.elementwise_kind<max_signed>} in %func
        : (!transform.any_op) -> !transform.any_op

    // ---- Step 5: 对 matmul 应用三级 Tiling ----
    // TB 层：核间分块
    %tb_m = transform.param.constant 64 : i64
    %tb_n = transform.param.constant 64 : i64

    %matmul_tb, %loop_tb_m, %loop_tb_n = transform.structured.tile_using_for %matmul [%tb_m, %tb_n]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 标记 TB 层为并行
    %true = transform.param.constant true : i1
    transform.annotate %loop_tb_m "ascendc.parallel" = %true
        : !transform.any_op, !transform.param<i1>
    transform.annotate %loop_tb_n "ascendc.parallel" = %true
        : !transform.any_op, !transform.param<i1>

    // Tb 层：核内分块
    %tb_m_inner = transform.param.constant 16 : i64
    %tb_n_inner = transform.param.constant 16 : i64

    %matmul_tb_inner, %loop_tb_m_inner, %loop_tb_n_inner = transform.structured.tile_using_for %matmul_tb [%tb_m_inner, %tb_n_inner]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // t_K 层：K 维度分块
    %t_k = transform.param.constant 32 : i64

    %matmul_t, %loop_t_k = transform.structured.tile_using_for %matmul_tb_inner [0, 0, %t_k]
        : (!transform.any_op, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op)

    // ---- Step 6: 标记执行单元 ----
    %cube_unit = transform.param.constant "AiCore.Cube" : !transform.any_param
    transform.annotate %matmul_t "ascendc.unit" = %cube_unit
        : !transform.any_op, !transform.any_param

    // ---- Step 7: 标记数据搬运 ----
    %prologue = transform.param.constant "lhs:GM->A1,rhs:GM->B1" : !transform.any_param
    %epilogue = transform.param.constant "acc:CO1->VECIN" : !transform.any_param
    transform.annotate %loop_t_k "ascendc.prologue" = %prologue
        : !transform.any_op, !transform.any_param
    transform.annotate %loop_t_k "ascendc.epilogue" = %epilogue
        : !transform.any_op, !transform.any_param

    // ---- Step 8: 对 add 和 relu 应用 Vector 标记 ----
    %vector_unit = transform.param.constant "AiCore.Vector" : !transform.any_param
    transform.annotate %add "ascendc.unit" = %vector_unit
        : !transform.any_op, !transform.any_param
    transform.annotate %relu "ascendc.unit" = %vector_unit
        : !transform.any_op, !transform.any_param

    transform.yield
  }

  // ----------------------------------------------------------
  // @canonicalize: 规范化变换
  // ----------------------------------------------------------
  transform.named_sequence @canonicalize(
      %root: !transform.any_op { transform.readonly }
  ) {
    // 应用规范化 pass
    transform.apply_patterns to %root {
      transform.apply_patterns.linalg.canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op

    transform.yield
  }

  // ----------------------------------------------------------
  // @optimize: 完整优化流程
  // ----------------------------------------------------------
  transform.named_sequence @optimize(
      %root: !transform.any_op { transform.readonly }
  ) {
    // 特化调度
    transform.include @specialize_fc_relu failures(propagate) (%root)
        : (!transform.any_op) -> ()

    // 规范化
    transform.include @canonicalize failures(propagate) (%root)
        : (!transform.any_op) -> ()

    transform.yield
  }

} // module
