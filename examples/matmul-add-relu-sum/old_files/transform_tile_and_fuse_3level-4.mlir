// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3 (AscendNPU 三级 Tiling) [v7]
// ============================================================
//
// ★ v7 核心改动：Add/ReLU 下沉到 for_Tb_N 内
//
//   v6 中 add/max fuse 进 for_TB_N，粒度是 TB_M×TB_N（粗）。
//   v7 改为 fuse 进 for_Tb_N，粒度是 Tb_M×Tb_N（细），原因:
//     1. 数据局部性: matmul(for_K) 结果在 L0C，紧接 add/max
//        避免写回 GM 再重新加载整个 TB tile
//     2. 硬件流水: CUBE Core(matmul) → Vector Core(add/max)
//        在同一 Tb 粒度上流水，双缓冲效率最高
//     3. 与目标 Step3 IR 一致: add/max 在 for_Tb_N 内、for_K 之后
//
// 关键顺序调整 (v6 vs v7):
//   v6: tile_max[TB] → fuse_add → fuse_matmul → tile_Tb_M → tile_Tb_N → tile_K
//   v7: tile_max[TB] → fuse_matmul_only → tile_Tb_M → tile_Tb_N → tile_K
//                    → fuse_add_into_Tb_N → fuse_max_into_Tb_N
//
// 最终循环结构:
//   scf.for TB_M (step=TB_M)        ← 核间 M
//     scf.for TB_N (step=TB_N)      ← 核间 N
//       scf.for Tb_M (step=Tb_M)    ← 核内 M，A_L1 复用
//         scf.for Tb_N (step=Tb_N)  ← 核内 N，B_L1 每次换
//           scf.for K (step=t_K)    ← CUBE 指令粒度
//             linalg.matmul
//           linalg.elementwise add  ← Vector Core，Tb 粒度 ✓
//           linalg.elementwise max  ← Vector Core，Tb 粒度 ✓
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
    //   顺序: TB_M, TB_N, Tb_M, Tb_N, t_K
    //   BlockArgument 是 SSA Value，类型必须是 !transform.any_op
    // ----------------------------------------------------------------
    %func_new, %TB_M, %TB_N, %Tb_M, %Tb_N, %t_K =
        transform.func.add_index_args %func, 5
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,   // TB_M: 核间 M 分块
                !transform.any_op,   // TB_N: 核间 N 分块
                !transform.any_op,   // Tb_M: 核内 M 分块
                !transform.any_op,   // Tb_N: 核内 N 分块
                !transform.any_op)   // t_K:  K CUBE 粒度

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
    // Step 4: tile_using_for max — TB 层
    //
    //   对 max 做 [TB_M, TB_N] tiling，产生最外两层 scf.for。
    //   与 Step1 完全一致，已验证可行。
    //
    //   返回:
    //     %tiled_max : for_TB_N 内 tiled 的 max
    //     %for_TB_M  : 外层 for (M 方向, step=TB_M)
    //     %for_TB_N  : 内层 for (N 方向, step=TB_N)
    // ----------------------------------------------------------------
    %tiled_max, %for_TB_M, %for_TB_N =
        transform.structured.tile_using_for %max
            tile_sizes [%TB_M, %TB_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 5: fuse matmul into for_TB_N
    //
    //   ★ v7 关键: 只 fuse matmul，不 fuse add。
    //     add/max 留到 for_Tb_N 建好后再 fuse（Step 9/10）。
    //
    //   fuse 后 matmul 位于 for_TB_N 内，操作 TB_M×TB_N 大小的子块。
    //   split_handle 取出真正的 matmul (#0)，#1/#2 是辅助 op。
    // ----------------------------------------------------------------
    %matmul_fused, %loop_matmul =
        transform.structured.fuse_into_containing_op %matmul into %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_split:3 = transform.split_handle %matmul_fused
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // %matmul_split#0: 真正的 matmul，位于 for_TB_N 内

    // ----------------------------------------------------------------
    // Step 6: tile_using_for matmul — Tb_M 层
    //
    //   对 for_TB_N 内的 matmul 做 [Tb_M, 0, 0] tiling。
    //   ★ 必须在 Tb_N 和 K tiling 之前做（从外到内）。
    //
    //   生成:
    //     for_TB_N {
    //       for_Tb_M (step=Tb_M) {   ← 新增
    //         linalg.matmul
    //       }
    //       max (TB粒度，暂未下沉)
    //     }
    //
    //   A_L1 的 extract_slice (Tb_M×K) 落在 for_Tb_M 内、
    //   for_Tb_N 外，实现 A-stationary 数据复用。
    // ----------------------------------------------------------------
    %tiled_matmul_Tb_M, %for_Tb_M =
        transform.structured.tile_using_for %matmul_split#0
            tile_sizes [%Tb_M, 0, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 7: tile_using_for matmul — Tb_N 层
    //
    //   ⚠️ 重新 match: 从 %for_Tb_M 内重新定位 matmul。
    //
    //   对 for_Tb_M 内的 matmul 做 [0, Tb_N, 0] tiling。
    //   B_L1 的 extract_slice (K×Tb_N) 落在 for_Tb_N 内。
    //
    //   ★ 保存 %for_Tb_N 句柄，Step 9/10 fuse add/max 时使用。
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
    // Step 8: tile_using_for matmul — K 层 (t_K 粒度)
    //
    //   ⚠️ 重新 match: 从 %for_Tb_N 内重新定位 matmul。
    //
    //   对 for_Tb_N 内的 matmul 做 [0, 0, t_K] tiling。
    //   对应 CUBE Core 单次指令处理的 K 粒度，操作 L0A/L0B/L0C。
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
    // Step 9: fuse add into for_Tb_N
    //
    //   ★ v7 核心: add 下沉到 for_Tb_N，而非 for_TB_N。
    //
    //   此时 for_Tb_N 内已有 for_K { matmul }，
    //   matmul 的结果（Tb_M×Tb_N tile）在 for_Tb_N 内可见，
    //   fuse_into_containing_op 可以找到切入点。
    //
    //   fuse 后结构:
    //     for_Tb_N {
    //       for_K { matmul }
    //       add (Tb_M×Tb_N 粒度)   ← ✓ Vector Core，紧跟 CUBE 结果
    //     }
    // ----------------------------------------------------------------
    %add_fused, %loop_add =
        transform.structured.fuse_into_containing_op %add into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 10: fuse max into for_Tb_N
    //
    //   max 消费 add 的结果，同样下沉到 for_Tb_N。
    //
    //   最终 for_Tb_N 内结构:
    //     for_Tb_N {
    //       for_K { matmul }         ← CUBE Core
    //       add (Tb_M×Tb_N)          ← Vector Core
    //       max (Tb_M×Tb_N)          ← Vector Core (ReLU)
    //     }
    // ----------------------------------------------------------------
    %max_fused, %loop_max =
        transform.structured.fuse_into_containing_op %max into %for_Tb_N
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 11: hoist_loop_invariant_subsets
    //
    //   由内向外提升，保证内层提升结果能被外层继续识别:
    //
    //   对 %for_Tb_N: 不依赖 iv_Tb_N 的切片提升到 for_Tb_M 内:
    //     - bias 的 Tb_M 级切片 (extract bias[iv_Tb_M, :][Tb_M, TB_N])
    //     - zero_tensor 的 Tb_M 级切片
    //
    //   对 %for_Tb_M: 不依赖 iv_Tb_M 的切片提升到 for_TB_N 内:
    //     - bias 的 TB 级切片 (extract bias[iv_TB_M, iv_TB_N][TB_M, TB_N])
    //     - zero_tensor 的 TB 级切片
    //     最终 bias_TB / zero_TB 在 for_TB_N 内只加载一次，
    //     Tb 循环内从 L1 Buffer 取子块，不再访问 GM。
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}
