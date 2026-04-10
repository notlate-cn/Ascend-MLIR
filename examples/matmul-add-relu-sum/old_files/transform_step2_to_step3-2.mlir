// ============================================================
// Transform Dialect 脚本: Step2 → Step3 (AscendNPU 三级 Tiling) [v2 Fixed]
// ============================================================
//
// 修复说明 (相对上一版本):
//   Fix 1: add_index_args 返回值类型
//           旧: -> (!transform.any_op, !transform.any_op, ...)
//           新: -> (!transform.any_op,       ← 变换后的新 func 句柄
//                   !transform.any_value,    ← TB_M BlockArgument
//                   !transform.any_value,    ← TB_N BlockArgument
//                   !transform.any_value,    ← Tb_M BlockArgument
//                   !transform.any_value)    ← Tb_N BlockArgument
//           新增参数是 func 的 BlockArgument (SSA Value 而非 Op)，
//           必须用 !transform.any_value 承接，否则类型验证失败。
//
//   Fix 2: tile_sizes 中动态 Value 的类型声明
//           tile_sizes 方括号内的动态 Value 在类型签名里对应
//           !transform.any_value，静态常量 0 不占类型签名位置。
//           写法:
//             tile_sizes [%val_M, %val_N, 0]
//                 : (!transform.any_op,    ← op 句柄
//                    !transform.any_value, ← val_M
//                    !transform.any_value) ← val_N
//
//   Fix 3: Step C/D 放弃复用 tile_using_forall 返回的 tiled op 句柄
//           改为在上一步产生的父循环内重新 match linalg.matmul，
//           规避句柄在 IR 重建后失效导致的 TilingInterface 错误。
// ============================================================

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.readonly}
  ) {

    // ----------------------------------------------------------------
    // 0. 获取目标 func
    // ----------------------------------------------------------------
    %func = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.any_op

    // ================================================================
    // Step A: add_index_args
    //
    //   接口:
    //     transform.func.add_index_args %func_handle, N
    //         : (!transform.any_op)
    //         -> (!transform.any_op,      ← 变换后新 func 句柄
    //             !transform.any_value,   ← 第1个新增参数的 BlockArgument
    //             ...共 N 个...)
    //
    //   ⚠️ 关键类型区分:
    //     - func 句柄本身是 Op，用 !transform.any_op
    //     - 新增的 BlockArgument 是 SSA Value，必须用 !transform.any_value
    //     混用会在 transform verifier 阶段报类型不匹配错误。
    //
    //   追加顺序 (func 签名末尾新增4个 index 参数):
    //     %new_TB_M : index    核间 M 分块大小
    //     %new_TB_N : index    核间 N 分块大小
    //     %new_Tb_M : index    核内 M 分块大小
    //     %new_Tb_N : index    核内 N 分块大小
    // ================================================================
    %func_new, %TB_M_arg, %TB_N_arg, %Tb_M_arg, %Tb_N_arg =
        transform.func.add_index_args %func, 4
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op,
                !transform.any_op)

    // ================================================================
    // Step B: tile_using_forall (TB 层 — 核间并行)
    //
    //   目标: func 内的 linalg.matmul
    //         matmul 实现了 TilingInterface，可直接作为 tiling 目标。
    //         Step2 中 matmul 虽已被两层 scf.for 包裹，但这里我们
    //         重新对 matmul 做 TB 级 tiling，forall 会作为 matmul 的
    //         新外层循环插入，原有的两层 scf.for 在 canonicalize 后
    //         会因为被替换而消除。
    //
    //   tile_sizes [%TB_M_arg, %TB_N_arg, 0]:
    //     维度0 (M): 步长 = TB_M (动态 Value)
    //     维度1 (N): 步长 = TB_N (动态 Value)
    //     维度2 (K): 0 = 不在 TB 层切 K 轴
    //
    //   类型签名: op 句柄 + 每个动态 Value 各占一个 !transform.any_value
    //
    //   生成结构:
    //     scf.forall (%iv_TB_M, %iv_TB_N) step(TB_M, TB_N)
    //       shared_outs(%out = ...) -> tensor<?x?xf32> {
    //       ... extract_slice A/B/C (TB 粒度) ...
    //       scf.for %iv_K step(t_K) { linalg.matmul }
    //       linalg.elementwise add
    //       linalg.elementwise max_signed
    //       in_parallel { tensor.parallel_insert_slice ... }
    //     }
    // ================================================================
    %matmul_0 = transform.structured.match ops{["linalg.matmul"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %forall_op, %tiled_matmul_TB =
        transform.structured.tile_using_forall %matmul_0
            tile_sizes [%TB_M_arg, %TB_N_arg, 0]
                : (!transform.any_op,
                   !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ================================================================
    // Step C: tile_using_for — Tb_M 层 (核内 M 方向分块)
    //
    //   ⚠️ 重新 match 而非复用 %tiled_matmul_TB:
    //     tile_using_forall 执行后会在内部重建 matmul op，
    //     %tiled_matmul_TB 句柄在 LLVM 17/18 中可能已被标记为
    //     replaced，再传给 tile_using_for 时框架无法从中解析
    //     TilingInterface，触发:
    //       "only ops implementing TilingInterface are supported"
    //     解决方案: 从 %forall_op 内部重新 match，拿到当前
    //     IR 状态中真实存活的 matmul 句柄。
    //
    //   tile_sizes [%Tb_M_arg, 0, 0]:
    //     维度0 (M): 步长 = Tb_M
    //     维度1 (N): 0 = 不切
    //     维度2 (K): 0 = 不切
    //
    //   生成新一层 scf.for (Tb_M 循环)，A_L1 的提取落在此层内、
    //   Tb_N 循环外，实现 A-stationary 数据复用。
    // ================================================================
    %matmul_1 = transform.structured.match ops{["linalg.matmul"]} in %forall_op
        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb_M, %for_Tb_M =
        transform.structured.tile_using_for %matmul_1
            tile_sizes [%Tb_M_arg, 0, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ================================================================
    // Step D: tile_using_for — Tb_N 层 (核内 N 方向分块)
    //
    //   ⚠️ 同 Step C，在 %for_Tb_M 内重新 match matmul。
    //
    //   tile_sizes [0, %Tb_N_arg, 0]:
    //     维度0 (M): 0 = 不切
    //     维度1 (N): 步长 = Tb_N
    //     维度2 (K): 0 = 不切
    //
    //   完成后最终循环嵌套:
    //     scf.forall (TB_M × TB_N)     ← 核间并行, Step B
    //       scf.for Tb_M               ← 核内 M,   Step C，A_L1 复用
    //         scf.for Tb_N             ← 核内 N,   Step D，B_L1 每次换
    //           scf.for K (t_K)        ← 原有 K 循环，CUBE 指令粒度
    //             linalg.matmul
    // ================================================================
    %matmul_2 = transform.structured.match ops{["linalg.matmul"]} in %for_Tb_M
        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb_N, %for_Tb_N =
        transform.structured.tile_using_for %matmul_2
            tile_sizes [0, %Tb_N_arg, 0]
                : (!transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ================================================================
    // Step E: hoist_loop_invariant_subsets
    //
    //   由内向外依次提升，保证内层提升结果能被外层继续识别。
    //
    //   对 %for_Tb_N (Tb_N 循环):
    //     将不依赖 %iv_Tb_N 的 tensor.extract_slice 提升到循环外:
    //       - bias 的 TB 级切片 (仅依赖 %iv_TB_M/%iv_TB_N，与 Tb_N 无关)
    //       - zero 的 TB 级切片 (同上)
    //     提升后 Tb_N 循环内从已加载的 L1 Buffer 取子块，不再访问 GM。
    //
    //   对 %for_Tb_M (Tb_M 循环):
    //     继续将不依赖 %iv_Tb_M 的切片提升到 forall 体内:
    //       - 经上步提升的 bias_TB/zero_TB 如不依赖 Tb_M 偏移则继续上提
    //     A_L1 依赖 %iv_Tb_M，保持原位不动。
    // ================================================================
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

    transform.yield
  }

} // end module

// ====================================================================
// 执行方式:
//   mlir-opt exp_output_step2_canonicalize.mlir \
//       --transform-interpreter=entry-point=__transform_main \
//       --allow-unregistered-dialect \
//       -o exp_output_step3_3-level.mlir
// ====================================================================
