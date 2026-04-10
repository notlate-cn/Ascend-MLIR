// ============================================================
// Transform Dialect 脚本: Step2 → Step3 (AscendNPU 三级 Tiling)
// ============================================================
//
// 输入 (Step2) 参数布局:
//   %arg0: tensor<?x?xf32>   A [M, K]
//   %arg1: tensor<?x?xf32>   B [K, N]
//   %arg2: tensor<?x?xf32>   bias [M, N]
//   %arg3: tensor<?x?xf32>   output [M, N]
//   %arg4: i64               旧 t_M  (Step2 的 M 步长，变换后作废)
//   %arg5: i64               旧 t_N  (Step2 的 N 步长，变换后作废)
//   %arg6: i64               旧 t_K  (Step2 的 K 步长，保留为 t_K)
//
// 输出 (Step3) 参数布局（新增参数追加到末尾）:
//   %arg4: i64               t_K  (保留原 K 步长语义)
//   %arg5: i64               TB_N (核间 N 分块, 新增)  ← add_index_args
//   %arg6: i64               TB_M (核间 M 分块, 新增)  ← add_index_args
//   %arg7: i64               Tb_M (核内 M 分块, 新增)  ← add_index_args
//   %arg8: i64               Tb_N (核内 N 分块, 新增)  ← add_index_args
//
// 变换流水线 (5步):
//   Step A: add_index_args        — 向 func 末尾追加 TB_M/TB_N/Tb_M/Tb_N 四个 index 参数
//   Step B: tile_using_forall     — 将最外两层 scf.for (M/N) 替换为 scf.forall (核间并行)
//                                    tile_sizes = [TB_M, TB_N] (来自新增的 index 参数)
//   Step C: tile_using_for (Tb_M) — 在 forall 内对 matmul 再做 M 方向 tiling
//                                    tile_sizes = [Tb_M, 0, 0]
//   Step D: tile_using_for (Tb_N) — 在上步产生的循环内再做 N 方向 tiling
//                                    tile_sizes = [0, Tb_N, 0]
//   Step E: hoist_loop_invariant_subsets
//                                 — 将 bias/zero 的 TB 级 extract_slice 提升到 forall 体内
//                                    Tb 循环外，消除循环内重复的 GM 访问
// ============================================================

// -------------------------------------------------------------------
// 注意: transform.sequence 在 MLIR upstream 已迁移至
//       transform.named_sequence，此处按社区最新接口书写。
//       若使用较旧工具链（<= LLVM 17），将 named_sequence 改回
//       transform.sequence failures(propagate) 即可。
// -------------------------------------------------------------------

module attributes {transform.with_named_sequence} {

  // ==================================================================
  // 主变换序列入口
  // ==================================================================
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
    //   向 @fc_relu 末尾追加 4 个 index 类型参数:
    //     新 %arg7_idx (TB_M), %arg8_idx (TB_N),
    //     新 %arg9_idx (Tb_M), %arg10_idx (Tb_N)
    //
    //   add_index_args 返回值依次对应新增参数在 func 体内的 BlockArgument。
    //   这些 BlockArgument 随后作为 tile_sizes 传给 tile_using_forall /
    //   tile_using_for，使 tile 大小真正来自运行时参数，而非编译期常量。
    // ================================================================
    %transformed, %TB_M_arg, %TB_N_arg, %Tb_M_arg, %Tb_N_arg =
        transform.func.add_index_args %func, 4 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // 参数追加顺序: TB_M, TB_N, Tb_M, Tb_N
    // 函数签名变为:
    //   @fc_relu(...原有7个参数...,
    //            %new0: index,  // TB_M
    //            %new1: index,  // TB_N
    //            %new2: index,  // Tb_M
    //            %new3: index)  // Tb_N

    // ================================================================
    // Step B: tile_using_forall  (TB 层 — 核间并行)
    //
    //   目标: linalg.matmul (Step2 中唯一的结构化 linalg op，
    //         外层已有两层 scf.for 驱动 M/N 方向)
    //
    //   效果: 将 matmul 最外两个 tile 循环替换为 scf.forall，
    //         生成:
    //           scf.forall (%iv_TB_M, %iv_TB_N) ... step (%TB_M, %TB_N)
    //             shared_outs(...)
    //           scf.forall.in_parallel {
    //             tensor.parallel_insert_slice ...
    //           }
    //
    //   tile_sizes: [TB_M_arg, TB_N_arg, 0]
    //     — 第三维 (K) 填 0 表示不在此层切分 K 轴
    //     — 前两维使用 Step A 新增的运行时 index 参数
    //
    //   mapping: #gpu.thread<x>, #gpu.thread<y> 仅作占位符语义，
    //            在 AscendNPU lowering 时会被替换为 AI Core 映射。
    //            若工具链不支持 gpu mapping，可省略 mapping 属性，
    //            forall 仍具备并行语义，由后端决定如何分发。
    // ================================================================
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %transformed
        : (!transform.any_op) -> !transform.any_op

    %forall_op, %tiled_matmul_TB =
        transform.structured.tile_using_forall %matmul
            tile_sizes [%TB_M_arg, %TB_N_arg, 0]
                : (!transform.any_op, !transform.any_op,
                   !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    // 此时 IR 结构:
    //   scf.forall (%iv_TB_M, %iv_TB_N) step(TB_M, TB_N) {   ← 新
    //     ... extract A/B/C TB slices ...
    //     scf.for %iv_K ... step(t_K) { linalg.matmul }      ← 原 K 循环
    //     linalg.elementwise add
    //     linalg.elementwise max
    //     in_parallel { parallel_insert_slice }
    //   }

    // ================================================================
    // Step C: tile_using_for — Tb_M 层 (核内 M 方向分块)
    //
    //   目标: Step B 产生的 %tiled_matmul_TB (forall 内的 matmul)
    //
    //   tile_sizes: [Tb_M_arg, 0, 0]
    //     — 仅切 M 轴，N/K 轴填 0
    //     — 生成一层新的 scf.for，步长 = Tb_M (运行时参数)
    //
    //   效果:
    //     scf.forall (...) {
    //       scf.for %iv_Tb_M = 0 to sz_TB_M step Tb_M {    ← 新
    //         ... extract A_L1 (sz_Tb_M × K) ...
    //         scf.for %iv_K { linalg.matmul }
    //       }
    //     }
    //
    //   A_L1 提取操作此时位于 Tb_M 循环内、Tb_N 循环外，
    //   天然实现"A 片在 Tb_N 方向复用"语义（A-stationary）。
    // ================================================================
    %tiled_matmul_Tb_M, %for_Tb_M =
        transform.structured.tile_using_for %tiled_matmul_TB
            tile_sizes [%Tb_M_arg, 0, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    // 此时 IR 结构:
    //   scf.forall (...) {
    //     scf.for %iv_Tb_M step(Tb_M) {                     ← Step C 新增
    //       scf.for %iv_K { linalg.matmul }
    //     }
    //   }

    // ================================================================
    // Step D: tile_using_for — Tb_N 层 (核内 N 方向分块)
    //
    //   目标: Step C 产生的 %tiled_matmul_Tb_M
    //
    //   tile_sizes: [0, Tb_N_arg, 0]
    //     — 仅切 N 轴，M/K 轴填 0
    //
    //   效果:
    //     scf.for %iv_Tb_M {
    //       A_L1 = extract A (sz_Tb_M × K)                 ← 保持不动
    //       scf.for %iv_Tb_N step(Tb_N) {                  ← Step D 新增
    //         B_L1 = extract B (K × sz_Tb_N)
    //         scf.for %iv_K { linalg.matmul }              ← t 层
    //       }
    //     }
    //
    //   此时三级循环嵌套结构与 Step3 目标 IR 完全吻合。
    // ================================================================
    %tiled_matmul_Tb_N, %for_Tb_N =
        transform.structured.tile_using_for %tiled_matmul_Tb_M
            tile_sizes [0, %Tb_N_arg, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    // ================================================================
    // Step E: hoist_loop_invariant_subsets  (切片提升 — 消除冗余 GM 访问)
    //
    //   目标: Tb_M 循环 (%for_Tb_M) 和 Tb_N 循环 (%for_Tb_N)
    //
    //   效果1 (对 %for_Tb_N):
    //     将 Tb_N 循环体内"仅依赖 %iv_TB_M / %iv_TB_N 而不依赖 %iv_Tb_N"
    //     的 tensor.extract_slice 提升到 Tb_N 循环外:
    //       — bias 的 TB 级切片 (bias_TB)
    //       — zero 的 TB 级切片 (zero_TB)
    //     这两个切片在所有 Tb_N 迭代中值不变，提升后每个 AI Core 只做
    //     一次 GM 读取，后续 Tb_N 循环内从 L1 Buffer (bias_TB/zero_TB) 取子块。
    //
    //   效果2 (对 %for_Tb_M):
    //     将 Tb_M 循环体内不随 %iv_Tb_M 变化的切片继续上提到 forall 体内，
    //     进一步减少不必要的重复提取。
    //
    //   注意: hoist_loop_invariant_subsets 会自动识别 loop-invariant 的
    //         tensor.extract_slice / tensor.insert_slice 对，并将其提升；
    //         不满足条件的切片（如依赖循环变量的 A_L1/B_L1）保持原位不动。
    // ================================================================
    transform.loop.hoist_loop_invariant_subsets %for_Tb_N
        : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %for_Tb_M
        : !transform.any_op

//    // ================================================================
//    // Step F: canonicalize (可选，整理产物)
//    //
//    //   清理 tile 过程中产生的冗余 affine.min / arith 表达式，
//    //   使输出 IR 与手写的 Step3 风格更接近。
//    // ================================================================
//    transform.apply_patterns.transform.erase_unnecessary_inputs %func
//        : !transform.any_op
//    transform.apply_patterns to %func {
//      transform.apply_patterns.canonicalization
//    } : !transform.any_op

    transform.yield
  } // end named_sequence @__transform_main

} // end module (transform)

// ====================================================================
// 补充说明
// ====================================================================
//
// 1. add_index_args 与 i64 参数的关系
//    Step2 的 %arg4/%arg5/%arg6 是 i64 类型，在 func 体内通过
//    arith.index_cast 转为 index 使用。
//    add_index_args 追加的是原生 index 类型参数，直接可作为
//    tile_using_for / tile_using_forall 的 tile_sizes 使用，
//    无需额外 cast。
//    若下游 ABI 要求统一 i64，可在 func 体入口处为新参数补充
//    一个 arith.index_cast，或在调用侧 cast 后传入。
//
// 2. Step B 的 mapping 属性
//    tile_using_forall 可携带 mapping 属性指定并行维度的硬件绑定:
//      mapping = [#gpu.thread<y>, #gpu.thread<x>]
//    AscendNPU 后端若定义了自定义 DeviceMappingAttr，将此处替换为
//    对应属性即可，变换逻辑本身不受影响。
//    当前脚本省略 mapping，forall 以抽象并行语义存在，
//    由后续 lower-to-ascend-core pass 决定分发策略。
//
// 3. 旧参数 %arg4 (旧 t_M) / %arg5 (旧 t_N) 的处理
//    Step2 的 %arg4/%arg5 在 tile_using_forall 完成后，
//    其对应的 arith.index_cast 和外层 scf.for 步长引用已被替换，
//    变为死代码。canonicalize pass (Step F) 会自动消除它们。
//    若需保持函数签名稳定，可在 Step F 后调用
//    transform.func.remove_dead_arguments 显式清理。
//
// 4. 执行方式
//    mlir-opt input_step2.mlir \
//        --transform-interpreter=entry-point=__transform_main \
//        --allow-unregistered-dialect \
//        -o output_step3.mlir
//
// ====================================================================
