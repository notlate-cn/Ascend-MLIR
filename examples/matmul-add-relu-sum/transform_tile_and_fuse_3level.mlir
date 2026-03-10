// 文件: transform_tile_and_fuse_3level.mlir (Transform Dialect 脚本)
// 函数: __transform_main
// 功能: 3级循环分块 + 融合 + 确定搬运时机 + 确定计算单元
// ============================================================
// 设计原则:
//   transform 脚本只携带调度策略，硬件拓扑知识集中在后续的自定义Pass(AscendCBufferPlacementPass)里。
//
//   携带的信息:
//     1. 循环结构: tile sizes (TB_M/TB_N/Tb_M/Tb_N/t_K) + 嵌套层次
//     2. 分核标注: ascendc.parallel = true
//                  标注 for_TB_M / for_TB_N 为分核循环
//                  后端 lowering 将其映射到多 AiCore 并行
//     3. 循环级搬运时机:
//          ascendc.prologue: 循环入口（或 AiCore 入口）执行的搬运
//          ascendc.epilogue: 循环出口（或 AiCore 出口）执行的搬运
//          格式: "角色:路径,角色:路径,..."（逗号分隔，支持多目标）
//          时机: 挂在哪层 for 就在那层入口/出口执行
//          粒度: 由该循环的 tile size 自然决定
//     4. Op执行单元
//          格式：ascendc.unit = "计算单元"
//          单元枚举：
//          AiCore.Cube: 矩阵乘法
//          AiCore.Vector: 向量加法/ReLU/累加
//          AiCpu: AiCpu执行单元
//
// 分核与搬运的关系:
//   for_TB_M × for_TB_N 共同构成分核空间（ascendc.parallel = true）
//   每个 AiCore 负责输出矩阵的一个 [TB_M × TB_N] 块，
//   对应唯一的 A[TB_M × K] 和 B[K × TB_N] 子块，互不重叠。
//   因此 GM->A1/B1 在分核后、Tb 循环前执行（挂在 for_TB_N.prologue），
//   每个 AiCore 独立搬运自己负责的数据，无重复搬运。
//
// 最终循环结构及 annotation 分布:
//
//   scf.for %TB_M {ascendc.parallel = true}
//     scf.for %TB_N {ascendc.parallel = true,
//                    prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN",
//                    epilogue = "result:VECOUT->GM"}
//       scf.for %Tb_M
//         scf.for %Tb_N
//           scf.for %K {prologue = "lhs:A1->A2,rhs:B1->B2",
//                        epilogue = "acc:CO1->VECIN"}
//             linalg.matmul          // unit 由 pass 推导: AiCore.Cube
//           linalg.elementwise add   // unit 由 pass 推导: AiCore.Vector
//           linalg.elementwise max   // unit 由 pass 推导: AiCore.Vector
//
// 搬运时机说明:
//   for_TB_N.prologue (GM->A1/B1/VECIN):
//     分核后每个 AiCore 的入口，搬运 TB_M×K / K×TB_N / TB_M×TB_N 粒度数据
//   for_TB_N.epilogue (VECOUT->GM):
//     每个 AiCore 完成全部 Tb 计算后写回，TB_M×TB_N 粒度
//   for_K.prologue (A1->A2 / B1->B2):
//     每次 K 迭代搬运 [Tb_M×t_K] / [t_K×Tb_N] 粒度的子块进 L0A/L0B
//   for_K.epilogue (CO1->VECIN):
//     K 轴所有迭代完成，CO1 是完整的 Tb_M×Tb_N 累加结果，
//     立刻 FixpipeOp 搬入 UB，供后续 add/max 消费
//
// AscendCBufferPlacementPass 推导规则:
//   Step A: 解析 prologue/epilogue，建立各层搬运任务表
//   Step B: 识别 ascendc.parallel = true 的循环为分核边界
//           prologue/epilogue 对应每个 AiCore 的入口/出口搬运
//   Step C: 按 op 类型推导执行单元
//             linalg.matmul      → AiCore.Cube  → lhs:A2, rhs:B2, out:CO1
//             linalg.elementwise → AiCore.Vector
//               ins 来自 CO1 epilogue 目标 → VECIN
//               ins 来自 GM prologue 目标  → VECIN
//               中间结果（有后续 Vector 消费者）→ VECCALC
//               最终输出（无后续 Vector 消费者）→ VECOUT
//   Step D: 插入显式搬运 op
//             GM->A1/B1:   ascendc.copy(GlobalTensor → LocalTensor<A1/B1>)
//             GM->VECIN:   ascendc.copy(GlobalTensor → LocalTensor<VECIN>)
//             A1->A2:      ascendc.copy(LocalTensor<A1> → LocalTensor<A2>)
//             B1->B2:      ascendc.copy(LocalTensor<B1> → LocalTensor<B2>)
//             CO1->VECIN:  ascendc.fixpipe(LocalTensor<CO1> → LocalTensor<VECIN>)
//             VECOUT->GM:  ascendc.copy(LocalTensor<VECOUT> → GlobalTensor)
// ============================================================

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}
  ) {

    // ----------------------------------------------------------------
    // Step 1: 匹配 func.func
    // ----------------------------------------------------------------
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op

    // ----------------------------------------------------------------
    // Step 2: add_index_args — 追加5个 index 参数
    //   顺序: TB_M, TB_N, Tb_M, Tb_N, t_K
    // ----------------------------------------------------------------
    %func_new, %TB_M, %TB_N, %Tb_M, %Tb_N, %t_K =
        transform.func.add_index_args %func, 5
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ----------------------------------------------------------------
    // Step 3: 匹配原始 linalg ops（在任何 tiling 之前）
    // ----------------------------------------------------------------
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // IR 顺序: #0=add, #1=max
    %add, %max = transform.split_handle %elementwise
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // 预定义 param 常量
    // ----------------------------------------------------------------

    // 分核标注
    %p_true = transform.param.constant true -> !transform.any_param

    // for_TB_N 搬运时机（分核后每个 AiCore 的入口/出口）:
    //   prologue: 搬运该 AiCore 负责的 A/B/bias 子块到 L1/UB
    //   epilogue: 将计算结果从 UB 写回 GM
    %p_TB_prologue = transform.param.constant
        "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"
        -> !transform.any_param
    %p_TB_epilogue = transform.param.constant
        "result:VECOUT->GM"
        -> !transform.any_param

    // for_K 搬运时机:
    //   prologue: 每次 K 迭代将 t_K 宽度的 A/B 子块从 L1 搬入 L0A/L0B
    //   epilogue: K 轴所有迭代完成，FixpipeOp 将完整累加结果从 L0C 搬入 UB
    %p_K_prologue = transform.param.constant
        "lhs:A1->A2,rhs:B1->B2"
        -> !transform.any_param
    %p_K_epilogue = transform.param.constant
        "acc:CO1->VECIN"
        -> !transform.any_param

    // Op 执行单元
    %p_cube   = transform.param.constant "AiCore.Cube"   -> !transform.any_param
    %p_vector = transform.param.constant "AiCore.Vector" -> !transform.any_param

    // ================================================================
    // 第一轮 TileAndFuse：TB 层
    // ================================================================

    // ----------------------------------------------------------------
    // Step 4: tile max [TB_M, TB_N] → for_TB_M / for_TB_N
    // ----------------------------------------------------------------
    %tiled_max_TB, %for_TB_M, %for_TB_N =
        transform.structured.tile_using_for %max
            tile_sizes [%TB_M, %TB_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 5: fuse add into for_TB_N
    // ----------------------------------------------------------------
    %add_fused_TB, %loop_add_TB =
        transform.structured.fuse_into_containing_op %add into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 6: fuse matmul into for_TB_N
    // ----------------------------------------------------------------
    %matmul_fused_TB, %loop_matmul_TB =
        transform.structured.fuse_into_containing_op %matmul into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_TB_split:3 = transform.split_handle %matmul_fused_TB
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ★ 分核标注: for_TB_M / for_TB_N 共同构成分核空间
    //   AscendCBufferPlacementPass 识别这两层为分核边界
    //   后端 lowering 将其映射到多 AiCore 并行执行
    transform.annotate %for_TB_M "ascendc.parallel"
        = %p_true : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.parallel"
        = %p_true : !transform.any_op, !transform.any_param

    // ★ 搬运时机: for_TB_N（分核后每个 AiCore 的入口/出口）
    transform.annotate %for_TB_N "ascendc.prologue"
        = %p_TB_prologue : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.epilogue"
        = %p_TB_epilogue : !transform.any_op, !transform.any_param

    // ================================================================
    // 第二轮 TileAndFuse：Tb 层
    // ================================================================

    // ----------------------------------------------------------------
    // Step 7: tile tiled_max_TB [Tb_M, Tb_N] → for_Tb_M / for_Tb_N
    // ----------------------------------------------------------------
    %tiled_max_Tb, %for_Tb_M, %for_Tb_N =
        transform.structured.tile_using_for %tiled_max_TB
            tile_sizes [%Tb_M, %Tb_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 8: fuse add_fused_TB into for_Tb_N
    // ----------------------------------------------------------------
    %add_fused_Tb, %loop_add_Tb =
        transform.structured.fuse_into_containing_op %add_fused_TB into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 9: fuse matmul into for_Tb_N
    // ----------------------------------------------------------------
    %matmul_fused_Tb, %loop_matmul_Tb =
        transform.structured.fuse_into_containing_op %matmul_TB_split#0 into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_Tb_split:3 = transform.split_handle %matmul_fused_Tb
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 10: tile matmul [0, 0, t_K] → for_K
    // ----------------------------------------------------------------
    %tiled_matmul_K, %for_K =
        transform.structured.tile_using_for %matmul_Tb_split#0
            tile_sizes [0, 0, %t_K]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_final = transform.structured.match ops{["linalg.matmul"]} in %for_K
        : (!transform.any_op) -> !transform.any_op
    transform.annotate %matmul_final "ascendc.unit"
        = %p_cube : !transform.any_op, !transform.any_param

    transform.annotate %add_fused_Tb "ascendc.unit"
        = %p_vector : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_Tb "ascendc.unit"
        = %p_vector : !transform.any_op, !transform.any_param

    // ★ 搬运时机: for_K
    //   prologue: 每次 K 迭代将 [Tb_M×t_K]/[t_K×Tb_N] 子块搬入 L0A/L0B
    //   epilogue: K 轴完成后 FixpipeOp CO1→VECIN，供 add/max 消费
    transform.annotate %for_K "ascendc.prologue"
        = %p_K_prologue : !transform.any_op, !transform.any_param
    transform.annotate %for_K "ascendc.epilogue"
        = %p_K_epilogue : !transform.any_op, !transform.any_param

    // ----------------------------------------------------------------
    // Step 11: hoist_loop_invariant_subsets
    //   由内向外提升循环不变切片:
    //   for_Tb_N: 提升不依赖 iv_Tb_N 的切片到 for_Tb_M 内
    //   for_Tb_M: 继续提升不依赖 iv_Tb_M 的切片到 for_TB_N 内
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}
