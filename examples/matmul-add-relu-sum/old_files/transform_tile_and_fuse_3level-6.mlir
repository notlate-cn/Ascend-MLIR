// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3 + TPosition Annotation
// [v9 - 在 transform 阶段打 buffer 语义 attribute]
// ============================================================
//
// ★ 设计原则 (v9 新增):
//   循环深度 ≠ buffer 语义。同一循环层内可能同时存在：
//     - matmul lhs  → A1/A2
//     - matmul rhs  → B1/B2
//     - matmul out  → CO1
//     - add ins     → VECIN
//     - add out     → VECOUT
//   必须在 transform 阶段（此时 op 语义最清晰）打 attribute 标注。
//
//   标注策略：每次 tile / fuse 完成后，对产出的 op 句柄立即
//   用 transform.annotate 打上 "ascendc.tposition_*" attribute：
//     ascendc.tposition_lhs  : lhs 操作数对应的 TPosition
//     ascendc.tposition_rhs  : rhs 操作数对应的 TPosition
//     ascendc.tposition_out  : output 操作数对应的 TPosition
//     ascendc.tposition_ins  : elementwise ins 对应的 TPosition
//     ascendc.core           : "CUBE" 或 "VECTOR"，标注执行引擎
//
//   后续 bufferize pass 只需读这些 attribute，
//   完全不依赖循环深度，语义准确且可扩展。
//
// TPosition 映射（对应 AscendC API TPosition 枚举）:
//   matmul TB层 lhs  → A1   (L1 Buffer, 存放 A 的 GM→L1 搬运结果)
//   matmul TB层 rhs  → B1   (L1 Buffer, 存放 B 的 GM→L1 搬运结果)
//   matmul Tb层 lhs  → A2   (L0A, 存放 A 的 L1→L0A 搬运结果)
//   matmul Tb层 rhs  → B2   (L0B, 存放 B 的 L1→L0B 搬运结果)
//   matmul 所有层 out→ CO1  (L0C, CUBE Core 输出累加缓冲)
//   add/max ins      → VECIN  (UB, Vector Core 输入)
//   add/max out      → VECOUT (UB, Vector Core 输出)
//
// 最终循环结构:
//   scf.for TB_M
//     scf.for TB_N
//       scf.for Tb_M
//         scf.for Tb_N
//           scf.for K (t_K)
//             linalg.matmul {ascendc.core="CUBE",
//                            ascendc.tposition_lhs="A2",
//                            ascendc.tposition_rhs="B2",
//                            ascendc.tposition_out="CO1"}
//           linalg.elementwise add {ascendc.core="VECTOR",
//                                   ascendc.tposition_ins="VECIN",
//                                   ascendc.tposition_out="VECOUT"}
//           linalg.elementwise max {ascendc.core="VECTOR",
//                                   ascendc.tposition_ins="VECIN",
//                                   ascendc.tposition_out="VECOUT"}
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
    //   TB_M, TB_N, Tb_M, Tb_N, t_K
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

    // ----------------------------------------------------------------
    // ★ Annotation A: TB 层 matmul 的 buffer 语义标注
    //
    //   此时 matmul 操作的是 TB_M×TB_N 大小的 tile，
    //   对应从 GM 搬运到 L1 Buffer 的数据：
    //     lhs (A) → A1 (L1 Buffer)
    //     rhs (B) → B1 (L1 Buffer)
    //     out (C) → CO1 (L0C，累加结果)
    //   执行引擎为 CUBE Core。
    //
    //   注：TB 层 matmul 是 fuse 后产生的中间 op，
    //       Tb 层会对它再次 tile，最终执行的是 Tb 层的 matmul。
    //       此处标注主要用于指导 TB 层 DataCopy 插入：
    //       GM→A1, GM→B1。
    // ----------------------------------------------------------------
    transform.annotate %matmul_TB_split#0 "ascendc.core" = "CUBE"
        : !transform.any_op
    transform.annotate %matmul_TB_split#0 "ascendc.tposition_lhs" = "A1"
        : !transform.any_op
    transform.annotate %matmul_TB_split#0 "ascendc.tposition_rhs" = "B1"
        : !transform.any_op
    transform.annotate %matmul_TB_split#0 "ascendc.tposition_out" = "CO1"
        : !transform.any_op

    // ----------------------------------------------------------------
    // ★ Annotation B: TB 层 add/max 的 buffer 语义标注
    //
    //   TB 层的 add/max 操作 TB_M×TB_N 大小的 tile，
    //   执行引擎为 VECTOR Core，操作 Unified Buffer (UB)：
    //     ins (matmul 结果, bias) → VECIN
    //     out                    → VECOUT
    //
    //   注：Tb 层 fuse 后 add/max 会下沉，此 TB 层标注
    //       主要记录 bias 的 GM→VECIN 搬运语义。
    // ----------------------------------------------------------------
    transform.annotate %add_fused_TB "ascendc.core" = "VECTOR"
        : !transform.any_op
    transform.annotate %add_fused_TB "ascendc.tposition_ins" = "VECIN"
        : !transform.any_op
    transform.annotate %add_fused_TB "ascendc.tposition_out" = "VECOUT"
        : !transform.any_op

    transform.annotate %tiled_max_TB "ascendc.core" = "VECTOR"
        : !transform.any_op
    transform.annotate %tiled_max_TB "ascendc.tposition_ins" = "VECIN"
        : !transform.any_op
    transform.annotate %tiled_max_TB "ascendc.tposition_out" = "VECOUT"
        : !transform.any_op

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
    // Step 9: fuse matmul_TB_split#0 into for_Tb_N
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

    // ----------------------------------------------------------------
    // ★ Annotation C: Tb 层（最终执行层）matmul 的精确 buffer 语义
    //
    //   这是最终真正执行 CUBE 指令的 matmul，操作 Tb_M×Tb_N tile：
    //     lhs (A slice) → A2  (L0A，从 L1 Buffer 搬入)
    //     rhs (B slice) → B2  (L0B，从 L1 Buffer 搬入)
    //     out (C)       → CO1 (L0C，CUBE 累加输出)
    //   执行引擎为 CUBE Core。
    //
    //   ★ 关键：覆盖之前 Annotation A 的标注，
    //     因为 tile 后这个 matmul 操作的是更小的 Tb_M×Tb_N 粒度，
    //     对应 L1→L0 的搬运语义（A2/B2），而非 GM→L1（A1/B1）。
    // ----------------------------------------------------------------
    %matmul_in_for_K = transform.structured.match ops{["linalg.matmul"]} in %for_K
        : (!transform.any_op) -> !transform.any_op

    transform.annotate %matmul_in_for_K "ascendc.core" = "CUBE"
        : !transform.any_op
    transform.annotate %matmul_in_for_K "ascendc.tposition_lhs" = "A2"
        : !transform.any_op
    transform.annotate %matmul_in_for_K "ascendc.tposition_rhs" = "B2"
        : !transform.any_op
    transform.annotate %matmul_in_for_K "ascendc.tposition_out" = "CO1"
        : !transform.any_op

    // ----------------------------------------------------------------
    // ★ Annotation D: Tb 层 add/max 的精确 buffer 语义
    //
    //   add/max 操作 Tb_M×Tb_N 大小的 tile，执行引擎为 VECTOR Core：
    //     add:
    //       ins[0] (matmul CO1 结果，经 CO1→UB 搬运) → VECIN
    //       ins[1] (bias slice，经 GM→UB 搬运)        → VECIN
    //       out                                       → VECOUT
    //     max (ReLU):
    //       ins[0] (add 结果)                         → VECIN
    //       ins[1] (zero tensor)                      → VECIN
    //       out                                       → VECOUT
    //
    //   VECCALC 用于 Vector 计算过程中的临时中间结果，
    //   由 ascir dialect lowering 时根据算子复杂度自动插入，
    //   此处不需要在 transform 层显式标注。
    // ----------------------------------------------------------------
    transform.annotate %add_fused_Tb "ascendc.core" = "VECTOR"
        : !transform.any_op
    transform.annotate %add_fused_Tb "ascendc.tposition_ins" = "VECIN"
        : !transform.any_op
    transform.annotate %add_fused_Tb "ascendc.tposition_out" = "VECOUT"
        : !transform.any_op

    transform.annotate %tiled_max_Tb "ascendc.core" = "VECTOR"
        : !transform.any_op
    transform.annotate %tiled_max_Tb "ascendc.tposition_ins" = "VECIN"
        : !transform.any_op
    transform.annotate %tiled_max_Tb "ascendc.tposition_out" = "VECOUT"
        : !transform.any_op

    // ----------------------------------------------------------------
    // ★ Annotation E: DataCopy 语义标注（搬运方向）
    //
    //   标注各层循环对应的数据搬运方向，指导后续 bufferize pass
    //   在正确位置插入 DataCopy + EnQue/DeQue：
    //     for_TB_M/N：GM→A1 (lhs), GM→B1 (rhs), GM→VECIN (bias)
    //     for_Tb_M/N：A1→A2 (lhs), B1→B2 (rhs)
    //     for_K 后：  CO1→VECIN (matmul结果进 Vector Core)
    //     for_Tb_N 后：VECOUT→GM (最终结果写回)
    //
    //   用 for 循环句柄打标，表示"进入此循环时需插入的搬运"。
    // ----------------------------------------------------------------
    transform.annotate %for_TB_N "ascendc.copy_in"  = "GM_to_L1"
        : !transform.any_op
    transform.annotate %for_Tb_N "ascendc.copy_in"  = "L1_to_L0"
        : !transform.any_op
    transform.annotate %for_Tb_N "ascendc.copy_out" = "L0C_to_UB"
        : !transform.any_op
    transform.annotate %for_TB_N "ascendc.copy_out" = "UB_to_GM"
        : !transform.any_op

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

// ====================================================================
// 后续 bufferize pass 使用约定：
//
// 读取 attribute 的规则:
//   1. 对每个 linalg op，读 "ascendc.tposition_lhs/rhs/ins/out"，
//      在 bufferize 时将对应的 alloc memref 转换为带该 TPosition
//      标注的 LocalTensor（或 GlobalTensor）。
//
//   2. 对每个 scf.for，读 "ascendc.copy_in/out"，
//      在循环入口/出口插入对应的 DataCopy + EnQue/DeQue。
//
//   3. "ascendc.core" = "CUBE" 的 op → lowering 到 ascir.mmad
//      "ascendc.core" = "VECTOR" 的 op → lowering 到 ascir.add/max
//
// 这样 bufferize pass 的逻辑变为：
//   "读 attribute → 分配对应 TPosition 的 buffer → 插入搬运"
//   完全不依赖循环深度，语义准确。
// ====================================================================
