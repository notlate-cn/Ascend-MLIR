// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3 (AscendNPU 三级 Tiling) [v8-final]
// ============================================================
//
// 问题根因:
//   fuse_into_containing_op 要求 producer(add/max) 的结果
//   必须被 container loop 内部的 tensor.extract_slice 引用。
//   matmul 经过多次 tile_using_for 重建后，add 的 ins[0]
//   仍指向 func 级别的原始 %matmul，与内层 for_Tb_N 没有
//   任何 def-use 连接，导致 fuse 找不到切入点。
//
// 正确策略: 完全倒序 TileAndFuse
//   每一层 tiling 以最终消费者(max)为起点，add/matmul 作为
//   producer 在每层 tiling 后立即 fuse，保持 def-use 链始终
//   连接到当前 container。
//
//   倒序流程:
//     1. tile max [TB_M, TB_N]        → for_TB_M / for_TB_N
//     2. fuse add  into for_TB_N      → add  进入 for_TB_N
//     3. fuse matmul into for_TB_N    → matmul 进入 for_TB_N
//
//     4. tile max [Tb_M, Tb_N]        → for_Tb_M / for_Tb_N (对 for_TB_N 内的 tiled_max 再次 tiling)
//     5. fuse add_fused  into for_Tb_N → add 下沉到 for_Tb_N
//     6. fuse matmul_fused into for_Tb_N → matmul 下沉到 for_Tb_N
//     7. tile matmul [0, 0, t_K]      → for_K, add/max 现在在 for_Tb_N 内、for_K 外  ──
//
//     8. hoist
//
// 最终循环结构:
//   scf.for TB_M (step=TB_M)
//     scf.for TB_N (step=TB_N)
//       scf.for Tb_M (step=Tb_M)
//         scf.for Tb_N (step=Tb_N)
//           scf.for K (step=t_K)
//             linalg.matmul          ← CUBE Core
//           linalg.elementwise add   ← Vector Core ✓
//           linalg.elementwise max   ← Vector Core ✓
//
// 执行方式:
//   mlir-opt fc_add_relu.mlir \
//       --transform-interpreter=entry-point=__transform_main \
//       --allow-unregistered-dialect \
//       -o output_step3.mlir
// ============================================================

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}
  ) {

    // ----------------------------------------------------------------
    // Step 1: 匹配 func.func
    // ----------------------------------------------------------------
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // ----------------------------------------------------------------
    // Step 2: add_index_args — 追加5个 index 参数
    //   顺序: TB_M, TB_N, Tb_M, Tb_N, t_K
    // ----------------------------------------------------------------
    %func_new, %TB_M, %TB_N, %Tb_M, %Tb_N, %t_K =
        transform.func.add_index_args %func, 5
            : (!transform.any_op)
            -> (!transform.any_op,   // 新的func函数
                !transform.any_op,   // TB_M
                !transform.any_op,   // TB_N
                !transform.any_op,   // Tb_M
                !transform.any_op,   // Tb_N
                !transform.any_op)   // t_K

    // ----------------------------------------------------------------
    // Step 3: 匹配原始 linalg ops (在任何 tiling 之前)
    // ----------------------------------------------------------------
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new : (!transform.any_op) -> !transform.any_op
    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %func_new : (!transform.any_op) -> !transform.any_op

    // IR 顺序: #0=add, #1=max
    %add, %max = transform.split_handle %elementwise : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ================================================================
    // 第一轮 TileAndFuse: TB 层
    // ================================================================

    // ----------------------------------------------------------------
    // Step 4: tile max [TB_M, TB_N] → for_TB_M / for_TB_N
    // ----------------------------------------------------------------
    %tiled_max_TB, %for_TB_M, %for_TB_N = transform.structured.tile_using_for %max tile_sizes [%TB_M, %TB_N] : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 5: fuse add into for_TB_N
    //   max←add 的 def-use 链在 for_TB_N 内可见，fuse 可行。
    // ----------------------------------------------------------------
    %add_fused_TB, %loop_add_TB = transform.structured.fuse_into_containing_op %add into %for_TB_N : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 6: fuse matmul into for_TB_N
    //   add←matmul 的 def-use 链在 for_TB_N 内可见，fuse 可行。
    //   split_handle 取出真正的 matmul (#0)。
    // ----------------------------------------------------------------
    %matmul_fused_TB, %loop_matmul_TB = transform.structured.fuse_into_containing_op %matmul into %for_TB_N : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_TB_split:3 = transform.split_handle %matmul_fused_TB : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // %matmul_TB_split#0: for_TB_N 内真正的 matmul

    // ================================================================
    // 第二轮 TileAndFuse: Tb 层 (新增，在 TB 层内部继续分块)
    //
    // 策略: 对 for_TB_N 内的 tiled_max_TB 再做一轮 [Tb_M, Tb_N] tiling，
    //       产生 for_Tb_M / for_Tb_N，然后把 add_fused_TB 和
    //       matmul_TB_split#0 依次 fuse 进 for_Tb_N。
    //       这样 add/max 天然落在 for_Tb_N 内，def-use 链始终保持连接。
    // ================================================================

    // ----------------------------------------------------------------
    // Step 7: tile tiled_max_TB [Tb_M, Tb_N] → for_Tb_M / for_Tb_N
    //
    //   对 for_TB_N 内的 %tiled_max_TB 再次做 [Tb_M, Tb_N] tiling。
    //   产生两层新的 scf.for，嵌套在 for_TB_N 内部。
    // ----------------------------------------------------------------
    %tiled_max_Tb, %for_Tb_M, %for_Tb_N = transform.structured.tile_using_for %tiled_max_TB tile_sizes [%Tb_M, %Tb_N] : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 8: fuse add_fused_TB into for_Tb_N
    //
    //   tiled_max_Tb 在 for_Tb_N 内，max←add 的 def-use 链
    //   通过 for_Tb_N 内的 extract_slice 连接到 add_fused_TB，
    //   fuse 可以找到切入点。
    //   add 下沉后位于 for_Tb_N 内，操作 Tb_M×Tb_N 粒度的子块。
    // ----------------------------------------------------------------
    %add_fused_Tb, %loop_add_Tb = transform.structured.fuse_into_containing_op %add_fused_TB into %for_Tb_N : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 9: fuse matmul_TB_split#0 into for_Tb_N
    //
    //   add_fused_Tb 在 for_Tb_N 内，add←matmul 的 def-use 链
    //   连接到 matmul_TB_split#0，fuse 可以找到切入点。
    //   matmul 下沉后位于 for_Tb_N 内，操作 Tb_M×Tb_N 粒度的子块。
    //   split_handle 取出真正的 matmul (#0)。
    // ----------------------------------------------------------------
    %matmul_fused_Tb, %loop_matmul_Tb = transform.structured.fuse_into_containing_op %matmul_TB_split#0 into %for_Tb_N : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_Tb_split:3 = transform.split_handle %matmul_fused_Tb : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // %matmul_Tb_split#0: for_Tb_N 内真正的 matmul

    // ================================================================
    // K 轴 tiling: 对 for_Tb_N 内的 matmul 做 K 轴分块
    // ================================================================

    // ----------------------------------------------------------------
    // Step 10: tile matmul [0, 0, t_K] → for_K
    //
    //   对 %matmul_Tb_split#0 做 K 轴 tiling，step=t_K。
    //   对应 CUBE Core 单次指令的 K 粒度，操作 L0A/L0B/L0C。
    //
    //   最终完整循环嵌套:
    //     scf.for TB_M (step=TB_M)        ← 核间 M
    //       scf.for TB_N (step=TB_N)      ← 核间 N
    //         scf.for Tb_M (step=Tb_M)    ← 核内 M，A_L1 复用
    //           scf.for Tb_N (step=Tb_N)  ← 核内 N，B_L1 每次换
    //             scf.for K (step=t_K)    ← CUBE 指令粒度
    //               linalg.matmul
    //             linalg.elementwise add  ← Vector Core ✓
    //             linalg.elementwise max  ← Vector Core ✓
    // ----------------------------------------------------------------
    %tiled_matmul_K, %for_K = transform.structured.tile_using_for %matmul_Tb_split#0 tile_sizes [0, 0, %t_K] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 11: hoist_loop_invariant_subsets
    //
    //   由内向外提升循环不变切片:
    //
    //   对 %for_Tb_N:
    //     不依赖 iv_Tb_N 的切片提升到 for_Tb_M 内:
    //       - bias 的切片 (依赖 iv_Tb_M 行偏移，不依赖 iv_Tb_N)
    //       - zero_tensor 的切片
    //
    //   对 %for_Tb_M:
    //     不依赖 iv_Tb_M 的切片继续提升到 for_TB_N 内:
    //       - bias 的 TB 级切片 (只依赖 iv_TB_M/iv_TB_N)
    //       - zero_tensor 的 TB 级切片
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M : !transform.any_op

    transform.yield
  }
}
