// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3 (AscendNPU 三级 Tiling) [v6-final]
// ============================================================
//
// 设计原则:
//   完全基于已验证可行的 Step1 tile_using_for + fuse_into_containing_op
//   风格，不使用 tile_using_forall，避免 fuse producer 失败的问题。
//   最外两层 scf.for (for_TB_M/for_TB_N) 保留，由后续 pass 转为
//   scf.forall 或直接由 AscendNPU 后端映射到 AI Core 并行。
//
// 关键顺序规则:
//   tiling 顺序必须从外到内: TB_M → TB_N(fuse) → Tb_M → Tb_N → K
//   不能先做 K 再做 Tb_M/Tb_N，否则 matmul 已被 K 包裹，
//   无法再对其做 M/N 方向的分块。
//
// 最终循环结构:
//   scf.for TB_M (step=TB_M)        ← 核间 M（可后续转 forall）
//     scf.for TB_N (step=TB_N)      ← 核间 N（可后续转 forall）
//       scf.for Tb_M (step=Tb_M)    ← 核内 M，A_L1 复用
//         scf.for Tb_N (step=Tb_N)  ← 核内 N，B_L1 每次换
//           scf.for K (step=t_K)    ← CUBE 单次 K 粒度
//             linalg.matmul
//           linalg.elementwise add  (fused)
//           linalg.elementwise max  (fused)
//
// 新增参数 (add_index_args 追加到 func 末尾，5个 index):
//   TB_M: 核间 M 分块大小
//   TB_N: 核间 N 分块大小
//   Tb_M: 核内 M 分块大小 (须满足 TB_M % Tb_M == 0)
//   Tb_N: 核内 N 分块大小 (须满足 TB_N % Tb_N == 0)
//   t_K:  K 轴 CUBE 单次处理粒度
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
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op

    // ----------------------------------------------------------------
    // Step 2: add_index_args — 追加5个 index 参数
    //   新增参数是 BlockArgument (SSA Value)，类型为 !transform.any_op
    // ----------------------------------------------------------------
    %func_new, %TB_M, %TB_N, %Tb_M, %Tb_N, %t_K =
        transform.func.add_index_args %func, 5
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,   // TB_M
                !transform.any_op,   // TB_N
                !transform.any_op,   // Tb_M
                !transform.any_op,   // Tb_N
                !transform.any_op)   // t_K

    // ----------------------------------------------------------------
    // Step 3: 匹配 linalg ops
    // ----------------------------------------------------------------
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %add, %max = transform.split_handle %elementwise
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 4: tile_using_for max — TB 层 (对齐 Step1 已验证路径)
    //
    //   对 max 做 [TB_M, TB_N] tiling，产生最外两层 scf.for。
    //   这条路径与 Step1 完全一致，已知可行。
    //
    //   返回:
    //     %tiled_max  : for_TB_N 内 tiled 的 max
    //     %for_TB_M   : 外层 for，step=TB_M
    //     %for_TB_N   : 内层 for，step=TB_N
    // ----------------------------------------------------------------
    %tiled_max, %for_TB_M, %for_TB_N =
        transform.structured.tile_using_for %max
            tile_sizes [%TB_M, %TB_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 5: fuse add into for_TB_N (Step1 已验证)
    // ----------------------------------------------------------------
    %add_fused, %loop_add =
        transform.structured.fuse_into_containing_op %add into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 6: fuse matmul into for_TB_N (Step1 已验证)
    // ----------------------------------------------------------------
    %matmul_fused, %loop_matmul =
        transform.structured.fuse_into_containing_op %matmul into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_split:3 = transform.split_handle %matmul_fused
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // %matmul_split#0: 真正的 matmul，现在位于 for_TB_N 内

    // ----------------------------------------------------------------
    // Step 7: tile_using_for matmul — Tb_M 层 (新增)
    //
    //   ★ 必须在 K tiling 之前做，否则 matmul 被 K 包裹后无法再
    //     对 M/N 维度分块。
    //
    //   对 %matmul_split#0 做 [Tb_M, 0, 0] tiling。
    //   dim0(M)=Tb_M, dim1(N)=0(不切), dim2(K)=0(不切)
    //
    //   生成结构:
    //     for_TB_N {
    //       for_Tb_M (step=Tb_M) {    ← 新增
    //         linalg.matmul
    //       }
    //       add / max
    //     }
    //
    //   A_L1 的 extract_slice (Tb_M×K) 落在 for_Tb_M 内、
    //   for_Tb_N 外，实现 A-stationary 复用。
    // ----------------------------------------------------------------
    %tiled_matmul_Tb_M, %for_Tb_M =
        transform.structured.tile_using_for %matmul_split#0
            tile_sizes [%Tb_M, 0, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 8: tile_using_for matmul — Tb_N 层 (新增)
    //
    //   ⚠️ 重新 match: 从 %for_Tb_M 内重新定位 matmul，
    //      避免使用可能失效的 %tiled_matmul_Tb_M 句柄。
    //
    //   对 for_Tb_M 内的 matmul 做 [0, Tb_N, 0] tiling。
    //   dim0(M)=0(不切), dim1(N)=Tb_N, dim2(K)=0(不切)
    //
    //   B_L1 的 extract_slice (K×Tb_N) 落在 for_Tb_N 内。
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
    // Step 9: tile_using_for matmul — K 层 (t_K 粒度，CUBE 指令)
    //
    //   ⚠️ 重新 match: 从 %for_Tb_N 内重新定位 matmul。
    //
    //   对 for_Tb_N 内的 matmul 做 [0, 0, t_K] tiling。
    //   dim2(K)=t_K，对应 CUBE Core 单次指令处理的 K 粒度。
    //
    //   最终完整循环嵌套:
    //     scf.for TB_M (step=TB_M)         ← 核间 M
    //       scf.for TB_N (step=TB_N)       ← 核间 N
    //         scf.for Tb_M (step=Tb_M)     ← 核内 M，A_L1 复用
    //           scf.for Tb_N (step=Tb_N)   ← 核内 N，B_L1 每次换
    //             scf.for K (step=t_K)     ← CUBE 指令粒度
    //               linalg.matmul
    //             linalg.elementwise add   (fused)
    //             linalg.elementwise max   (fused)
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
    //   由内向外依次提升，保证内层提升结果能被外层继续识别。
    //
    //   for_Tb_N 内不依赖 iv_Tb_N 的切片 → 提升到 for_Tb_M 内：
    //     - bias 的 TB 级切片 (extract_slice bias[TB_M×TB_N])
    //     - zero_tensor 的 TB 级切片
    //
    //   for_Tb_M 内不依赖 iv_Tb_M 的切片 → 继续提升到 for_TB_N 内：
    //     - 上步提升后的 bias_TB / zero_TB（若不依赖 iv_Tb_M）
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}
