// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3 + TPosition Annotation
// [v10-final]
//
// 语法说明 (LLVM 21.1.8):
//   - add_index_args 返回值类型: any_op (不是 any_value)
//   - transform.param.constant 用 -> 而不是 :
//   - transform.annotate 语法:
//       transform.annotate %target "key" = %param
//           : !transform.any_op, !transform.any_param
//
// Annotation 设计原则:
//   两类 attribute，职责严格分离：
//     ascendc.tposition_* : 描述 operand 的存储位置 (TPosition 枚举值)
//     ascendc.copy_*      : 描述搬运动作 (仅在需要搬运时存在，已就位则不打)
//     ascendc.core        : 执行引擎 ("CUBE" 或 "VECTOR")
//
// TPosition 映射:
//   A1   = L1 Buffer (lhs 从 GM 搬入)
//   B1   = L1 Buffer (rhs 从 GM 搬入)
//   A2   = L0A (lhs 从 L1 搬入)
//   B2   = L0B (rhs 从 L1 搬入)
//   CO1  = L0C (matmul 累加输出)
//   VECIN  = UB (Vector Core 输入)
//   VECOUT = UB (Vector Core 输出)
//
// 搬运时机:
//   for_TB_N 入口: GM→A1 (DataCopy lhs), GM→B1 (DataCopy rhs),
//                  GM→VECIN (DataCopy bias)
//   for_Tb_N 入口: A1→A2 (DataCopy), B1→B2 (DataCopy)
//   for_Tb_N 出口: CO1→VECIN (FixpipeOp，K轴累加完毕后)
//   for_TB_N 出口: VECOUT→GM (DataCopy 写回)
//
// 最终循环结构及各 op 的 attribute:
//   scf.for %TB_M
//     scf.for %TB_N  {copy_in="GM_to_L1", copy_out="VECOUT_to_GM"}
//       scf.for %Tb_M
//         scf.for %Tb_N  {copy_in="L1_to_L0", copy_out="CO1_to_VECIN"}
//           scf.for %K
//             linalg.matmul {core=CUBE,
//                            tposition_lhs=A2, tposition_rhs=B2,
//                            tposition_out=CO1}
//           linalg.elementwise add {core=VECTOR,
//                                   tposition_ins0=VECIN,  // matmul结果，已就位
//                                   tposition_ins1=VECIN,  // bias
//                                   tposition_out=VECOUT,
//                                   copy_ins1=GM_to_VECIN} // bias需DataCopy
//           linalg.elementwise max {core=VECTOR,
//                                   tposition_ins0=VECIN,  // add结果，已就位
//                                   tposition_ins1=VECIN,  // zero，见注释
//                                   tposition_out=VECOUT}
//   注: max 的 ins[1](zero tensor) 在 lowering 阶段替换为
//       AscendC Maxs 标量指令（与立即数0.0比较），无需DataCopy
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
    //   注: LLVM 21 中返回值类型均为 !transform.any_op
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
    // 预定义所有 param 常量
    //   注: LLVM 21 中 transform.param.constant 使用 -> 而不是 :
    // ----------------------------------------------------------------

    // core 类型
    %p_CUBE   = transform.param.constant "CUBE"   -> !transform.any_param
    %p_VECTOR = transform.param.constant "VECTOR" -> !transform.any_param

    // TPosition（存储位置）
    %p_A1     = transform.param.constant "A1"     -> !transform.any_param
    %p_B1     = transform.param.constant "B1"     -> !transform.any_param
    %p_A2     = transform.param.constant "A2"     -> !transform.any_param
    %p_B2     = transform.param.constant "B2"     -> !transform.any_param
    %p_CO1    = transform.param.constant "CO1"    -> !transform.any_param
    %p_VECIN  = transform.param.constant "VECIN"  -> !transform.any_param
    %p_VECCALC  = transform.param.constant "VECCALC"  -> !transform.any_param
    %p_VECOUT = transform.param.constant "VECOUT" -> !transform.any_param

    // 搬运动作（仅在需要搬运时使用）
    %p_GM_to_L1       = transform.param.constant "GM_to_L1"       -> !transform.any_param
    %p_L1_to_L0       = transform.param.constant "L1_to_L0"       -> !transform.any_param
    %p_CO1_to_VECIN   = transform.param.constant "CO1_to_VECIN"   -> !transform.any_param
    %p_VECOUT_to_GM   = transform.param.constant "VECOUT_to_GM"   -> !transform.any_param
    %p_GM_to_VECIN    = transform.param.constant "GM_to_VECIN"    -> !transform.any_param

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

    // ★ Annotation: for_TB_N 循环的搬运语义
    //   入口: GM→L1 (为 lhs/rhs 准备 A1/B1，为 bias 准备 VECIN)
    //   出口: VECOUT→GM (将最终结果写回 Global Memory)
    transform.annotate %for_TB_N "ascendc.copy_in"
        = %p_GM_to_L1 : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.copy_out"
        = %p_VECOUT_to_GM : !transform.any_op, !transform.any_param

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

    // ★ Annotation: for_Tb_N 循环的搬运语义
    //   入口: L1→L0 (A1→A2, B1→B2，为 CUBE 准备 L0A/L0B)
    //   出口: CO1→VECIN (FixpipeOp，K轴累加完毕，将结果从L0C搬入UB)
    transform.annotate %for_Tb_N "ascendc.copy_in"
        = %p_L1_to_L0 : !transform.any_op, !transform.any_param
    transform.annotate %for_Tb_N "ascendc.copy_out"
        = %p_CO1_to_VECIN : !transform.any_op, !transform.any_param

    // ★ Annotation: 最终执行层 matmul (for_K 内)
    //   位置: lhs=A2(L0A), rhs=B2(L0B), out=CO1(L0C)
    //   无 copy_* : operand 在循环入口已由 for_Tb_N.copy_in 就位
    %matmul_final = transform.structured.match ops{["linalg.matmul"]} in %for_K
        : (!transform.any_op) -> !transform.any_op

    transform.annotate %matmul_final "ascendc.core"
        = %p_CUBE : !transform.any_op, !transform.any_param
    transform.annotate %matmul_final "ascendc.tposition_lhs"
        = %p_A2 : !transform.any_op, !transform.any_param
    transform.annotate %matmul_final "ascendc.tposition_rhs"
        = %p_B2 : !transform.any_op, !transform.any_param
    transform.annotate %matmul_final "ascendc.tposition_out"
        = %p_CO1 : !transform.any_op, !transform.any_param

    // ★ Annotation: add op
    //   ins[0]: matmul 结果，for_Tb_N.copy_out (FixpipeOp) 完成后
    //           已在 UB(VECIN)，无需额外搬运，不打 copy_ins0
    //   ins[1]: bias，尚在 GM，需要 DataCopy GM→VECIN，打 copy_ins1
    //   out:    结果写入 UB(VECCALC)
    transform.annotate %add_fused_Tb "ascendc.core"
        = %p_VECTOR : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_Tb "ascendc.tposition_ins0"
        = %p_VECIN : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_Tb "ascendc.tposition_ins1"
        = %p_VECIN : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_Tb "ascendc.tposition_out"
        = %p_VECCALC : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_Tb "ascendc.copy_ins1"
        = %p_GM_to_VECIN : !transform.any_op, !transform.any_param
    // copy_ins0 不存在：ins[0] 已由 for_Tb_N.copy_out 就位

    // ★ Annotation: max op (ReLU)
    //   ins[0]: add 结果，已在 UB(VECCALC)，无需搬运
    //   ins[1]: zero tensor，lowering 阶段替换为 AscendC Maxs 标量指令
    //           (与立即数 0.0 比较)，不需要实际的 buffer，不打 copy_ins1
    //   out:    结果写入 UB(VECOUT)
    transform.annotate %tiled_max_Tb "ascendc.core"
        = %p_VECTOR : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_Tb "ascendc.tposition_ins0"
        = %p_VECCALC : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_Tb "ascendc.tposition_ins1"
        = %p_VECCALC : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_Tb "ascendc.tposition_out"
        = %p_VECOUT : !transform.any_op, !transform.any_param
    // copy_ins0 不存在：ins[0] 是 add 的输出，已在 UB
    // copy_ins1 不存在：ins[1] zero tensor 在 lowering 时消除

    // ----------------------------------------------------------------
    // Step 11: hoist_loop_invariant_subsets
    //   由内向外，提升 bias/zero 的不变切片：
    //   for_Tb_N → 提升不依赖 iv_Tb_N 的切片到 for_Tb_M 内
    //   for_Tb_M → 继续提升不依赖 iv_Tb_M 的切片到 for_TB_N 内
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}

// ====================================================================
// Bufferize pass 使用约定
// ====================================================================
//
// 读 ascendc.tposition_* → 为对应 operand 分配指定 TPosition 的 buffer
// 读 ascendc.copy_*      → 在对应位置插入搬运指令:
//   "GM_to_L1"     → DataCopy (GlobalTensor → LocalTensor<A1/B1>)
//   "L1_to_L0"     → DataCopy (LocalTensor<A1> → LocalTensor<A2>,
//                               LocalTensor<B1> → LocalTensor<B2>)
//   "CO1_to_VECIN" → FixpipeOp (LocalTensor<CO1> → LocalTensor<VECIN>)
//   "GM_to_VECIN"  → DataCopy (GlobalTensor → LocalTensor<VECIN>)
//   "VECOUT_to_GM" → DataCopy (LocalTensor<VECOUT> → GlobalTensor)
//
// 读 ascendc.core:
//   "CUBE"   → lowering 到 ascir.mmad
//   "VECTOR" → lowering 到 ascir.add / ascir.max(scalar 0.0)
// ====================================================================
