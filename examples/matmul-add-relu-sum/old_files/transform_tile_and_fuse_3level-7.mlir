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
    // Step 3: 匹配原始 linalg ops
    // ----------------------------------------------------------------
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %add, %max = transform.split_handle %elementwise
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // ★ 预先定义所有需要用到的 string 常量参数
    //   transform.annotate 的值必须通过 param handle 传入，
    //   不能直接写字符串字面量。
    // ----------------------------------------------------------------
    %p_CUBE    = transform.param.constant "CUBE"   -> !transform.any_param
    %p_VECTOR  = transform.param.constant "VECTOR" -> !transform.any_param
    %p_A1      = transform.param.constant "A1"     -> !transform.any_param
    %p_B1      = transform.param.constant "B1"     -> !transform.any_param
    %p_A2      = transform.param.constant "A2"     -> !transform.any_param
    %p_B2      = transform.param.constant "B2"     -> !transform.any_param
    %p_CO1     = transform.param.constant "CO1"    -> !transform.any_param
    %p_VECIN   = transform.param.constant "VECIN"  -> !transform.any_param
    %p_VECOUT  = transform.param.constant "VECOUT" -> !transform.any_param
    %p_GM_L1   = transform.param.constant "GM_to_L1"  -> !transform.any_param
    %p_L1_L0   = transform.param.constant "L1_to_L0"  -> !transform.any_param
    %p_L0C_UB  = transform.param.constant "L0C_to_UB" -> !transform.any_param
    %p_UB_GM   = transform.param.constant "UB_to_GM"  -> !transform.any_param

    // ================================================================
    // 第一轮 TileAndFuse：TB 层
    // ================================================================

    // Step 4: tile max [TB_M, TB_N] → for_TB_M / for_TB_N
    %tiled_max_TB, %for_TB_M, %for_TB_N =
        transform.structured.tile_using_for %max
            tile_sizes [%TB_M, %TB_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Step 5: fuse add into for_TB_N
    %add_fused_TB, %loop_add_TB =
        transform.structured.fuse_into_containing_op %add into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // Step 6: fuse matmul into for_TB_N
    %matmul_fused_TB, %loop_matmul_TB =
        transform.structured.fuse_into_containing_op %matmul into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_TB_split:3 = transform.split_handle %matmul_fused_TB
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ★ Annotation A: TB 层 matmul — GM→L1 搬运语义
    //   lhs→A1(L1), rhs→B1(L1), out→CO1(L0C)
    transform.annotate %matmul_TB_split#0 "ascendc.core"
        = %p_CUBE : !transform.any_op, !transform.any_param
    transform.annotate %matmul_TB_split#0 "ascendc.tposition_lhs"
        = %p_A1 : !transform.any_op, !transform.any_param
    transform.annotate %matmul_TB_split#0 "ascendc.tposition_rhs"
        = %p_B1 : !transform.any_op, !transform.any_param
    transform.annotate %matmul_TB_split#0 "ascendc.tposition_out"
        = %p_CO1 : !transform.any_op, !transform.any_param

    // ★ Annotation B: TB 层 add/max — GM→UB 搬运语义
    transform.annotate %add_fused_TB "ascendc.core"
        = %p_VECTOR : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_TB "ascendc.tposition_ins"
        = %p_VECIN : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_TB "ascendc.tposition_out"
        = %p_VECOUT : !transform.any_op, !transform.any_param

    transform.annotate %tiled_max_TB "ascendc.core"
        = %p_VECTOR : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_TB "ascendc.tposition_ins"
        = %p_VECIN : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_TB "ascendc.tposition_out"
        = %p_VECOUT : !transform.any_op, !transform.any_param

    // ★ Annotation E (loop): for_TB_N 搬运方向
    transform.annotate %for_TB_N "ascendc.copy_in"
        = %p_GM_L1 : !transform.any_op, !transform.any_param
    transform.annotate %for_TB_N "ascendc.copy_out"
        = %p_UB_GM : !transform.any_op, !transform.any_param

    // ================================================================
    // 第二轮 TileAndFuse：Tb 层
    // ================================================================

    // Step 7: tile tiled_max_TB [Tb_M, Tb_N] → for_Tb_M / for_Tb_N
    %tiled_max_Tb, %for_Tb_M, %for_Tb_N =
        transform.structured.tile_using_for %tiled_max_TB
            tile_sizes [%Tb_M, %Tb_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Step 8: fuse add_fused_TB into for_Tb_N
    %add_fused_Tb, %loop_add_Tb =
        transform.structured.fuse_into_containing_op %add_fused_TB into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // Step 9: fuse matmul into for_Tb_N
    %matmul_fused_Tb, %loop_matmul_Tb =
        transform.structured.fuse_into_containing_op %matmul_TB_split#0 into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_Tb_split:3 = transform.split_handle %matmul_fused_Tb
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Step 10: tile matmul [0, 0, t_K] → for_K
    %tiled_matmul_K, %for_K =
        transform.structured.tile_using_for %matmul_Tb_split#0
            tile_sizes [0, 0, %t_K]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // 在 transform 脚本的 Step 10 之后补充:
    %p_CO1_to_VECIN = transform.param.constant "CO1_to_VECIN"
        -> !transform.any_param
    %p_GM_to_VECIN  = transform.param.constant "GM_to_VECIN"
        -> !transform.any_param

    // add 的 ins[0] (matmul结果): CO1 → VECIN (FixpipeOp 搬运)
    transform.annotate %add_fused_Tb "ascendc.tposition_ins0"
        = %p_CO1_to_VECIN : !transform.any_op, !transform.any_param

    // add 的 ins[1] (bias): GM → VECIN (DataCopy 搬运)
    transform.annotate %add_fused_Tb "ascendc.tposition_ins1"
        = %p_GM_to_VECIN : !transform.any_op, !transform.any_param


    // ★ Annotation C: Tb 层最终执行 matmul — L1→L0 搬运语义
    //   ★ 覆盖 Annotation A：此层操作 Tb_M×Tb_N 子块，
    //     lhs→A2(L0A), rhs→B2(L0B), out→CO1(L0C)
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

    // ★ Annotation D: Tb 层 add/max — UB 计算语义
    transform.annotate %add_fused_Tb "ascendc.core"
        = %p_VECTOR : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_Tb "ascendc.tposition_ins"
        = %p_VECIN : !transform.any_op, !transform.any_param
    transform.annotate %add_fused_Tb "ascendc.tposition_out"
        = %p_VECOUT : !transform.any_op, !transform.any_param

    transform.annotate %tiled_max_Tb "ascendc.core"
        = %p_VECTOR : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_Tb "ascendc.tposition_ins"
        = %p_VECIN : !transform.any_op, !transform.any_param
    transform.annotate %tiled_max_Tb "ascendc.tposition_out"
        = %p_VECOUT : !transform.any_op, !transform.any_param

    // ★ Annotation E (loop): for_Tb_N 搬运方向
    transform.annotate %for_Tb_N "ascendc.copy_in"
        = %p_L1_L0 : !transform.any_op, !transform.any_param
    transform.annotate %for_Tb_N "ascendc.copy_out"
        = %p_L0C_UB : !transform.any_op, !transform.any_param

    // ----------------------------------------------------------------
    // Step 11: hoist_loop_invariant_subsets
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}
