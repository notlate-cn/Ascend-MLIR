// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3
// [v12-final]
//
// 语法 (LLVM 21.1.8):
//   - add_index_args 返回值类型均为 !transform.any_op
//   - transform.param.constant 使用 -> 而不是 :
//   - transform.annotate 语法:
//       transform.annotate %target "key" = %param
//           : !transform.any_op, !transform.any_param
//
// 设计原则:
//   transform 脚本只携带两类信息，硬件拓扑知识不在此处:
//     1. 循环结构: tile sizes + 嵌套层次
//     2. 调度策略:
//        a. 循环级搬运时机: ascendc.prologue / ascendc.epilogue
//           挂在哪层 for 就在那层循环入口/出口执行
//           粒度由该循环的 tile size 自然决定
//           格式: "角色:路径,角色:路径,..."（逗号分隔，支持多目标）
//        b. 执行单元: ascendc.unit
//           对 AiCore 内固定单元的 op 可省略（由 pass 按 op 类型推导）
//           对有选择空间的 op（如可选 AiCPU）必须显式标注
//
// AscendCBufferPlacementPass 负责（硬件知识集中在此）:
//   1. 读 ascendc.prologue/epilogue → 确定各层 buffer 的分配位置和搬运指令
//   2. 读 ascendc.unit（或按 op 类型推导）→ 确定片上计算单元
//   3. def-use 链分析 → 区分 UB 内 VECIN/VECCALC/VECOUT
//   4. 插入显式 ascendc.copy / ascendc.fixpipe op
//   5. 清除所有 ascendc.* annotation
//
// 最终循环结构及 annotation 分布:
//
//   scf.for %TB_M
//     scf.for %TB_N
//       {prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"}
//       {epilogue = "result:VECOUT->GM"}
//       scf.for %Tb_M
//         scf.for %Tb_N
//           {prologue = "lhs:A1->A2,rhs:B1->B2"}
//           scf.for %K
//             {epilogue = "acc:CO1->VECIN"}  ← K轴完成后 FixpipeOp
//             linalg.matmul
//             // ascendc.unit 省略，pass 按 op 类型推导为 AiCore.Cube
//           linalg.elementwise add
//             // ascendc.unit 省略，pass 按 op 类型推导为 AiCore.Vector
//           linalg.elementwise max
//             // ascendc.unit 省略，pass 按 op 类型推导为 AiCore.Vector
//
// 说明:
//   CO1->VECIN (FixpipeOp) 挂在 for_K.epilogue:
//     K 轴所有迭代完成后 CO1 才是完整累加结果，此时立刻 FixpipeOp
//     挂在 for_Tb_N.epilogue 会导致时机错误（add/max 已执行完）
//
//   result:VECOUT->GM 挂在 for_TB_N.epilogue 而非 for_TB_M.epilogue:
//     每个 TB_M×TB_N tile 计算完立刻写回 GM
//     UB 空间有限（910B 每 AiCore 256KB），不能攒整行再写回
//     实际写回由 for_Tb_N 内逐 Tb tile 完成，for_TB_N.epilogue
//     作为该 TB tile 写回完成的语义标记
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

    // for_TB_N 搬运时机:
    //   prologue（循环入口，TB_M×TB_N 粒度）:
    //     lhs:GM->A1   DataCopy A 矩阵 TB tile → L1 Buffer
    //     rhs:GM->B1   DataCopy B 矩阵 TB tile → L1 Buffer
    //     bias:GM->VECIN DataCopy bias TB tile → UB
    //   epilogue（循环出口，TB_M×TB_N 粒度）:
    //     result:VECOUT->GM  DataCopy 计算结果 → GM
    %p_TB_prologue = transform.param.constant
        "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"
        -> !transform.any_param
    %p_TB_epilogue = transform.param.constant
        "result:VECOUT->GM"
        -> !transform.any_param

    // for_Tb_N 搬运时机:
    //   prologue（循环入口，Tb_M×Tb_N 粒度）:
    //     lhs:A1->A2   DataCopy A 子块 L1→L0A
    //     rhs:B1->B2   DataCopy B 子块 L1→L0B
    %p_Tb_prologue = transform.param.constant
        "lhs:A1->A2,rhs:B1->B2"
        -> !transform.any_param

    // for_K 搬运时机:
    //   epilogue（K轴所有迭代完成后）:
    //     acc:CO1->VECIN  FixpipeOp L0C 完整累加结果 → UB
    //                     此后 add 直接消费 UB 中的结果，无需额外搬运
    %p_K_epilogue = transform.param.constant
        "acc:CO1->VECIN"
        -> !transform.any_param

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
    // #0: for_TB_N 内真正的 matmul

    // ★ Annotation: for_TB_N 的搬运时机
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
    // #0: for_Tb_N 内真正的 matmul

    // ----------------------------------------------------------------
    // Step 10: tile matmul [0, 0, t_K] → for_K
    // ----------------------------------------------------------------
    %tiled_matmul_K, %for_K =
        transform.structured.tile_using_for %matmul_Tb_split#0
            tile_sizes [0, 0, %t_K]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ★ Annotation: for_Tb_N 的搬运时机
    transform.annotate %for_Tb_N "ascendc.prologue"
        = %p_Tb_prologue : !transform.any_op, !transform.any_param

    // ★ Annotation: for_K 的搬运时机
    //   epilogue: K轴完成后立刻 FixpipeOp CO1→VECIN
    //   此时 add 在 for_K 之外、for_Tb_N 之内，
    //   FixpipeOp 完成后 add 直接消费 VECIN，时机精确
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

// ====================================================================
// AscendCBufferPlacementPass 推导规则（供实现参考）
// ====================================================================
//
// Step A: 解析各层循环的 prologue/epilogue，建立搬运任务表
//
//   for_TB_N.prologue → lhs:GM->A1, rhs:GM->B1, bias:GM->VECIN
//   for_TB_N.epilogue → result:VECOUT->GM
//   for_Tb_N.prologue → lhs:A1->A2, rhs:B1->B2
//   for_K.epilogue    → acc:CO1->VECIN (FixpipeOp)
//
// Step B: 按 op 类型推导执行单元（无 ascendc.unit 时的默认规则）
//   linalg.matmul      → AiCore.Cube
//   linalg.elementwise → AiCore.Vector
//   linalg.fill        → AiCore.Vector
//   （如有 ascendc.unit annotation 则以 annotation 为准）
//
// Step C: 按执行单元推导 operand memspace
//   AiCore.Cube (matmul):
//     operand(0) lhs → 最内层 prologue 中 lhs 的目标: A2 (L0A)
//     operand(1) rhs → 最内层 prologue 中 rhs 的目标: B2 (L0B)
//     operand(2) out → 固定为 CO1 (L0C)
//   AiCore.Vector (add/max):
//     ins 来自 CO1 的 epilogue 目标 → VECIN
//     ins 来自 GM 的 prologue 目标  → VECIN
//     out 有后续 Vector 消费者      → VECCALC（UB 内中转）
//     out 无后续 Vector 消费者      → VECOUT（准备写回 GM）
//
// Step D: 插入显式搬运 op
//   GM->A1/B1:   ascendc.copy(GlobalTensor → LocalTensor<A1/B1>)
//   A1->A2/B2:   ascendc.copy(LocalTensor<A1/B1> → LocalTensor<A2/B2>)
//   CO1->VECIN:  ascendc.fixpipe(LocalTensor<CO1> → LocalTensor<VECIN>)
//   GM->VECIN:   ascendc.copy(GlobalTensor → LocalTensor<VECIN>)
//   VECOUT->GM:  ascendc.copy(LocalTensor<VECOUT> → GlobalTensor)
//
// Step E: 清除所有 ascendc.* annotation
// ====================================================================
