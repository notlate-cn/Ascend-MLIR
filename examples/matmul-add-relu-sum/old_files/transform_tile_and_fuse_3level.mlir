// ============================================================
// Transform Dialect 脚本: fc_add_relu → Step3 (AscendNPU 三级 Tiling)
// ============================================================
//
// 输入: fc_add_relu.mlir (Step1, 无任何 for 包裹的原始计算图)
//   linalg.matmul
//   linalg.elementwise add
//   linalg.elementwise max_signed
//
// 目标结构 (Step3):
//   scf.forall (TB_M × TB_N)    ← 核间并行, add_index_args 新增 TB_M/TB_N
//     scf.for Tb_M              ← 核内 M, add_index_args 新增 Tb_M
//       scf.for Tb_N            ← 核内 N, add_index_args 新增 Tb_N
//         scf.for K (t_K)       ← K reduction, add_index_args 已有 t_K
//           linalg.matmul
//         linalg.elementwise add   (fused)
//         linalg.elementwise max   (fused)
//
// 参数布局 (共7个, 对应 Step1 原有4个tensor + 新增3个变为新增5个):
//   原有:
//     %lhs, %rhs, %bias, %output  : tensor 参数 (位置0~3)
//   新增 index 参数 (add_index_args 追加到末尾):
//     %TB_M  : index   核间 M 分块  (位置4)
//     %TB_N  : index   核间 N 分块  (位置5)
//     %Tb_M  : index   核内 M 分块  (位置6)
//     %Tb_N  : index   核内 N 分块  (位置7)
//     %t_K   : index   K 粒度       (位置8)
//
// 变换策略 (与 Step1 transform 的关键区别):
//   Step1 transform 对 max 做 tile_using_for [M, N]，产生两层 scf.for。
//   本脚本改为对 max 做 tile_using_forall [TB_M, TB_N]，直接产生
//   scf.forall 作为最外层；然后在 forall 内对 fused 后的 matmul
//   依次做 tile_using_for [Tb_M, 0] 和 [0, Tb_N]，产生两层 scf.for。
//   K 轴 tile 与 Step1 完全一致。
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
    //   返回值类型:
    //     第1个: !transform.any_op   — 变换后新 func 句柄
    //     后5个: !transform.any_op — 各 BlockArgument (SSA Value)
    //   追加顺序: TB_M, TB_N, Tb_M, Tb_N, t_K
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
    //   注意: match 在 %func_new 内进行，拿到 IR 当前真实存活的句柄
    // ----------------------------------------------------------------
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // split_handle: 把 add 和 max 分开
    // elementwise 按 IR 顺序返回: #0=add, #1=max
    %add, %max = transform.split_handle %elementwise
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 4: tile_using_forall — TB 层 (核间并行)
    //
    //   对 linalg.max 做 TB 级 tiling，产生最外层 scf.forall。
    //   tile_sizes [%TB_M, %TB_N]: 切 M(dim0) 和 N(dim1)，
    //   elementwise op 只有两个空间维度，无需填 0。
    //
    //   生成结构:
    //     scf.forall (%iv_TB_M, %iv_TB_N) step(TB_M, TB_N)
    //       shared_outs(%out = ...) {
    //       linalg.max (TB tile)
    //       in_parallel { parallel_insert_slice }
    //     }
    // ----------------------------------------------------------------
    %forall_op, %tiled_max_TB =
        transform.structured.tile_using_forall %max
            tile_sizes [%TB_M, %TB_N]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 5: fuse_into_containing_op — 将 add fuse 进 forall
    //
    //   将 linalg.add fuse 到 %forall_op 内部，
    //   使 add 与 max 在同一个 TB tile 内计算。
    //   fuse_into_containing_op 会自动在 forall 内生成
    //   对应的 extract_slice / insert_slice。
    // ----------------------------------------------------------------
    %add_fused, %add_fused_loop =
        transform.structured.fuse_into_containing_op %add into %forall_op
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 6: fuse_into_containing_op — 将 matmul fuse 进 forall
    //
    //   matmul fuse 后在 forall 内部，返回多个句柄（含辅助 matmul）。
    //   用 split_handle 取出真正的 matmul (#0)。
    // ----------------------------------------------------------------
    %matmul_fused, %matmul_fused_loop =
        transform.structured.fuse_into_containing_op %matmul into %forall_op
            : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    %matmul_fused_split:3 = transform.split_handle %matmul_fused
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // #0: 真正的 linalg.matmul (位于 forall 内)
    // #1/#2: 用于维度推断的辅助 matmul，canonicalize 后消除

    // ----------------------------------------------------------------
    // Step 7: tile_using_for — Tb_M 层 (核内 M 方向分块)
    //
    //   对 forall 内的 matmul (#0) 做 Tb_M 分块。
    //   tile_sizes [%Tb_M, 0, 0]: 仅切 M(dim0)，N/K 填 0。
    //
    //   生成:
    //     scf.forall (...) {
    //       scf.for %iv_Tb_M step(Tb_M) {   ← 新增
    //         linalg.matmul (Tb_M × N 子块)
    //       }
    //     }
    //   A_L1 的 extract_slice (Tb_M × K) 天然落在此层内、
    //   Tb_N 循环外，实现 A-stationary 复用。
    // ----------------------------------------------------------------
    %tiled_matmul_Tb_M, %for_Tb_M =
        transform.structured.tile_using_for %matmul_fused_split#0
            tile_sizes [%Tb_M, 0, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 8: tile_using_for — Tb_N 层 (核内 N 方向分块)
    //
    //   ⚠️ 重新 match: 从 %for_Tb_M 内部重新定位 matmul，
    //      避免使用可能已失效的 %tiled_matmul_Tb_M 句柄。
    //
    //   tile_sizes [0, %Tb_N, 0]: 仅切 N(dim1)，M/K 填 0。
    //
    //   生成:
    //     scf.for Tb_M {
    //       A_L1 = extract A[Tb_M × K]     ← 此层提取，Tb_N 方向复用
    //       scf.for %iv_Tb_N step(Tb_N) {  ← 新增
    //         B_L1 = extract B[K × Tb_N]
    //         linalg.matmul (Tb_M × Tb_N 子块)
    //       }
    //     }
    // ----------------------------------------------------------------
    %matmul_for_Tb_N = transform.structured.match ops{["linalg.matmul"]} in %for_Tb_M
        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb_N, %for_Tb_N =
        transform.structured.tile_using_for %matmul_for_Tb_N
            tile_sizes [0, %Tb_N, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 9: tile_using_for — t_K 层 (K 轴 reduction)
    //
    //   ⚠️ 重新 match: 从 %for_Tb_N 内部重新定位 matmul。
    //
    //   tile_sizes [0, 0, %t_K]: 仅切 K(dim2)，M/N 填 0。
    //   对应 CUBE Core 单次指令的 K 粒度，操作 L0A/L0B/L0C。
    //
    //   最终循环嵌套:
    //     scf.forall (TB_M × TB_N)      ← 核间并行
    //       scf.for Tb_M                ← 核内 M，A_L1 复用
    //         scf.for Tb_N              ← 核内 N，B_L1 每次换
    //           scf.for K (t_K)         ← CUBE 指令粒度
    //             linalg.matmul
    //           linalg.elementwise add  ← fused
    //           linalg.elementwise max  ← fused
    // ----------------------------------------------------------------
    %matmul_for_K = transform.structured.match ops{["linalg.matmul"]} in %for_Tb_N
        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_K, %for_K =
        transform.structured.tile_using_for %matmul_for_K
            tile_sizes [0, 0, %t_K]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------------
    // Step 10: hoist_loop_invariant_subsets — 切片提升
    //
    //   由内向外依次提升，保证内层提升结果能被外层继续识别。
    //
    //   对 %for_Tb_N:
    //     提升不依赖 %iv_Tb_N 的切片到 Tb_N 循环外 (Tb_M 循环内):
    //       - bias 的 TB 级切片
    //       - zero_tensor 的 TB 级切片
    //
    //   对 %for_Tb_M:
    //     继续将不依赖 %iv_Tb_M 的切片提升到 forall 体内。
    // ----------------------------------------------------------------
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }
}
