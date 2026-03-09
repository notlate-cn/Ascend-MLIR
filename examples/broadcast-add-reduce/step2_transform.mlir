// ============================================================
// STAGE 2: Transform Dialect Tiling - 使用变换方言进行分块
//
// 本阶段使用 Transform Dialect 描述分块策略：
//   - 输入: step1_fused.mlir 的单个 linalg.generic
//   - 迭代器类型: ["parallel", "reduction"]
//     * d0 = M (Parallel) - 可并行维度，用于核间分发
//     * d1 = N (Reduction) - 归约维度，完整遍历
//
// Tiling 结构 (仅对 Parallel 轴 d0 进行三级切分):
//   - TB 层: 核间并行，每核负责 TB_M 行 (标记 ascendc.parallel)
//   - Tb 层: UB (Unified Buffer) 批次，每批 Tb_M 行 (搬运粒度)
//   - d1 层: 归约轴完整遍历 N，使用向量化规约
//
// 函数签名扩展: 追加两个 index 参数 TB_M, Tb_M
//
// RUN: afir-opt --transform-interpreter %s | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

// 广播映射: 只访问 d0 轴 (用于输入 A 和输出 E)
#broadcast_map = affine_map<(d0, d1) -> (d0)>
// 完整访问映射: 访问 d0,d1 (用于输入 B)
#full_access_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  // ==========================================================
  // 主计算函数: 广播加法归约
  // ==========================================================
  func.func @broadcast_add_reducesum(
      %input_a: tensor<?xf16>,      // 输入 A [M]
      %input_b: tensor<?x?xf16>     // 输入 B [M,N]
  ) -> tensor<?xf16> {

    // 常量定义
    %zero = arith.constant 0.000000e+00 : f16
    %idx_0 = arith.constant 0 : index

    // 获取 M 维度大小
    %dim_m = tensor.dim %input_a, %idx_0 : tensor<?xf16>

    // 初始化输出张量
    %empty_output = tensor.empty(%dim_m) : tensor<?xf16>
    %init_output = linalg.fill ins(%zero : f16)
                   outs(%empty_output : tensor<?xf16>) -> tensor<?xf16>

    // 融合算子: Broadcast + Add + ReduceSum
    %output = linalg.generic {
      indexing_maps = [#broadcast_map, #full_access_map, #broadcast_map],
      iterator_types = ["parallel", "reduction"]
    } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
      outs(%init_output : tensor<?xf16>) {
    ^bb0(%a_value: f16, %b_value: f16, %accumulator: f16):
      // A[d0] + B[d0,d1]
      %sum = arith.addf %a_value, %b_value : f16
      // 累加
      %new_accumulator = arith.addf %accumulator, %sum : f16
      linalg.yield %new_accumulator : f16
    } -> tensor<?xf16>

    return %output : tensor<?xf16>
  }

  // ==========================================================
  // Transform 调度脚本: 定义分块策略
  // ==========================================================
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 1: 匹配 func.func，追加 2 个 index 参数 ----
    // 参数: TB_M (核间分块大小), Tb_M (UB 批次大小)
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ---- Step 2: 匹配 linalg.generic ----
    %generic = transform.structured.match ops{["linalg.generic"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // ---- Step 3: TB 层切分 (沿 d0=Parallel 轴，d1=Reduction 不切) ----
    // tile_sizes [%TB_M, 0]: 0 表示不切归约轴
    %tiled_tb, %loop_tb =
        transform.structured.tile_using_for %generic
            tile_sizes [%tb_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // 标记 TB 循环为核间并行 (映射到 AiCore 的 BlockIdx)
    %true_param = transform.param.constant true -> !transform.any_param
    transform.annotate %loop_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    // ---- Step 4: Tb 层切分 (沿 d0，UB 批次粒度) ----
    %tiled_tb_inner, %loop_tb_inner =
        transform.structured.tile_using_for %tiled_tb
            tile_sizes [%tb_inner_m_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // 标记 prologue (数据从 GM 搬运到 VECIN)
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    // 标记 epilogue (数据从 VECOUT 写回 GM)
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %loop_tb_inner "ascendc.prologue"
        = %prologue_param : !transform.any_op, !transform.any_param
    transform.annotate %loop_tb_inner "ascendc.epilogue"
        = %epilogue_param : !transform.any_op, !transform.any_param

    // ---- Step 5: 标注 Vector 执行单元 ----
    %vector_unit_param = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_tb_inner "ascendc.unit"
        = %vector_unit_param : !transform.any_op, !transform.any_param

    // ---- Step 6: 提升循环不变切片 ----
    // 将不依赖于 tb_inner 循环变量的 slice 提到 loop_tb 内
    transform.loop.hoist_loop_invariant_subsets %loop_tb_inner : !transform.any_op

    transform.yield
  }
}
