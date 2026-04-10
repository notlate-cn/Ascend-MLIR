// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3 (AscendNPU 三级 Tiling) [v5]
// ============================================================
//
// ★ v5 核心修正:
//   错误根因: tile_using_forall 作用于 elementwise op (max) 时，
//             框架不会为其 ins 输入生成 extract_slice，
//             导致后续 fuse_into_containing_op 找不到 producer
//             的切入点，报 "could not find next producer to fuse"。
//
//   正确做法: 对 matmul 做 tile_using_forall。
//     matmul 有独立的 outs tensor，tiling 后 forall 内会生成：
//       extract_slice A  (ins[0])
//       extract_slice B  (ins[1])
//       extract_slice C  (outs[0])  ← add/max 的结果最终写入此处
//     add 和 max 作为 matmul 结果的消费者，可以被
//     fuse_into_containing_op 顺序 fuse 进 forall。
//
// 变换流水线:
//   Step 2: add_index_args         新增 TB_M/TB_N/Tb_M/Tb_N/t_K
//   Step 4: tile_using_forall      对 matmul 做 TB 级 tiling → forall
//   Step 5: fuse add into forall   add fuse 进 forall
//   Step 6: fuse max into forall   max fuse 进 forall
//   Step 7: tile_using_for Tb_M    对 forall 内 matmul 切 M → for_Tb_M
//   Step 8: tile_using_for Tb_N    对 for_Tb_M 内 matmul 切 N → for_Tb_N
//   Step 9: tile_using_for t_K     对 for_Tb_N 内 matmul 切 K → for_K
//   Step 10: hoist                 提升 bias/zero 不变切片
//
// 执行方式:
//   mlir-opt fc_add_relu.mlir \
//       --transform-interpreter=entry-point=__transform_main \
//       --allow-unregistered-dialect \
//       -o exp_output_step3_3-level.mlir
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
    //   新增参数是 BlockArgument (SSA Value)，类型必须是
    //   !transform.any_op，不能用 !transform.any_op。
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
    // Step 3: 匹配 linalg ops
    // ----------------------------------------------------------------
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // IR 顺序: #0=add, #1=max
    %add, %max = transform.split_handle %elementwise
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 4: tile_using_forall — TB 层，对 matmul 做 TB 级 tiling
    //
    //   ★ 关键: 必须对 matmul 而非 elementwise op 做 tile_using_forall。
    //     matmul 有独立的 outs tensor，tiling 后 forall 内会生成：
    //       %es_A = tensor.extract_slice %lhs  [iv_M, 0][TB_M, K]
    //       %es_B = tensor.extract_slice %rhs  [0, iv_N][K,    TB_N]
    //       %es_C = tensor.extract_slice %out  [iv_M, iv_N][TB_M, TB_N]
    //       %res  = linalg.matmul ins(%es_A, %es_B) outs(%es_C)
    //       in_parallel { parallel_insert_slice %res into %out ... }
    //
    //     add 消费 matmul 的结果 %res，max 消费 add 的结果，
    //     fuse_into_containing_op 能沿 def-use 链找到切入点。
    //
    //   tile_sizes [%TB_M, %TB_N, 0]:
    //     dim0 (M): 步长 TB_M
    //     dim1 (N): 步长 TB_N
    //     dim2 (K): 0 = 不在 TB 层切 K (K 轴留给 t_K 层处理)
    // ----------------------------------------------------------------
    %forall_op, %tiled_matmul_TB =
        transform.structured.tile_using_forall %matmul
            tile_sizes [%TB_M, %TB_N, 0]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 5: fuse add into forall
    //
    //   add 消费 matmul 的结果，fuse_into_containing_op 会：
    //     1. 在 forall 内找到 matmul 结果的 use（in_parallel 块
    //        的 parallel_insert_slice，或直接使用）
    //     2. 将 add 的计算移入 forall，操作对应的 TB tile
    //     3. 生成 bias 的 extract_slice（[iv_M, iv_N][TB_M, TB_N]）
    //
    //   container 传 %forall_op（loop handle），不是 %tiled_matmul_TB。
    // ----------------------------------------------------------------
    %add_fused, %add_fused_loop =
        transform.structured.fuse_into_containing_op %add into %forall_op
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 6: fuse max into forall
    //
    //   max 消费 add 的结果，继续 fuse 进 forall。
    //   zero_tensor / linalg.fill 作为 max 的第二个输入，
    //   框架会自动处理其在 forall 内的 slice。
    // ----------------------------------------------------------------
    %max_fused, %max_fused_loop =
        transform.structured.fuse_into_containing_op %max into %forall_op
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 7: tile_using_for — Tb_M 层 (核内 M 方向分块)
    //
    //   ⚠️ 重新 match: tile_using_forall 后 %tiled_matmul_TB 句柄
    //      可能已失效，从 %forall_op 内重新定位 matmul。
    //
    //   tile_sizes [%Tb_M, 0, 0]: 仅切 M 轴。
    //   A_L1 的 extract_slice (Tb_M 行 × K 列) 落在此层循环内、
    //   Tb_N 循环外，实现 A-stationary 数据复用。
    // ----------------------------------------------------------------
    %matmul_in_forall = transform.structured.match ops{["linalg.matmul"]} in %forall_op
        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb_M, %for_Tb_M =
        transform.structured.tile_using_for %matmul_in_forall
            tile_sizes [%Tb_M, 0, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 8: tile_using_for — Tb_N 层 (核内 N 方向分块)
    //
    //   ⚠️ 重新 match: 从 %for_Tb_M 内重新定位 matmul。
    //
    //   tile_sizes [0, %Tb_N, 0]: 仅切 N 轴。
    //   B_L1 的 extract_slice (K 行 × Tb_N 列) 落在此层循环内。
    // ----------------------------------------------------------------
    %matmul_in_Tb_M = transform.structured.match ops{["linalg.matmul"]} in %for_Tb_M
        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb_N, %for_Tb_N =
        transform.structured.tile_using_for %matmul_in_Tb_M
            tile_sizes [0, %Tb_N, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 9: tile_using_for — t_K 层 (K 轴 reduction，CUBE 粒度)
    //
    //   ⚠️ 重新 match: 从 %for_Tb_N 内重新定位 matmul。
    //
    //   tile_sizes [0, 0, %t_K]: 仅切 K 轴。
    //
    //   最终循环嵌套:
    //     scf.forall (TB_M × TB_N)     ← 核间并行
    //       scf.for Tb_M               ← 核内 M，A 片复用
    //         scf.for Tb_N             ← 核内 N，B 片每次换
    //           scf.for K (t_K)        ← CUBE 单次指令粒度
    //             linalg.matmul
    //           linalg.elementwise add  (fused，Tb_M×Tb_N tile)
    //           linalg.elementwise max  (fused，Tb_M×Tb_N tile)
    // ----------------------------------------------------------------
    %matmul_in_Tb_N = transform.structured.match ops{["linalg.matmul"]} in %for_Tb_N
        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_K, %for_K =
        transform.structured.tile_using_for %matmul_in_Tb_N
            tile_sizes [0, 0, %t_K]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 10: hoist_loop_invariant_subsets
    //
    //   由内向外: 先 Tb_N，再 Tb_M。
    //   提升目标 (不依赖对应循环变量的切片):
    //     - bias 的 TB 级切片 → 提升到 Tb_N 外，进而提升到 forall 内
    //     - zero_tensor 的 TB 级切片 → 同上
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}
