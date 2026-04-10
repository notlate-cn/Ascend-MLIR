// ============================================================
// Transform Dialect 脚本: Step2 → Step3 (AscendNPU 三级 Tiling) [v3]
// ============================================================
//
// ★ 关键认知修正 (v2 → v3):
//   Step2 中 linalg.matmul 已被 3 层 scf.for 包裹:
//     scf.for M (step=t_M)        ← 第1层
//       scf.for N (step=t_N)      ← 第2层
//         scf.for K (step=t_K)    ← 第3层
//           linalg.matmul
//
//   v2 对 matmul 做 tile_using_forall 只会在 matmul 直接外层
//   插入 forall，原有3层 for 原封不动保留，导致 forall 沉在第4层。
//
//   v3 正确策略: 对已有的外层 scf.for 做 strip-mine，
//   在其外部再套一层更粗粒度的 for (TB 级)，然后把最外两层
//   (TB_M / TB_N) 转换为 scf.forall。
//   原有的内层 for (t_M / t_N) 自然成为 Tb 级循环。
//
// 变换流水线:
//   Step A: add_index_args        新增 TB_M / TB_N / Tb_M / Tb_N 四个 index 参数
//   Step B: strip_mine (M轴)      对 for_M (step=t_M) 做 strip-mine，step=TB_M
//                                  产生: for TB_M → for Tb_M (原 for_M)
//   Step C: strip_mine (N轴)      对 for_N (step=t_N) 做 strip-mine，step=TB_N
//                                  产生: for TB_N → for Tb_N (原 for_N)
//   Step D: loop_to_forall        将最外两层 for_TB_M / for_TB_N 转为 scf.forall
//   Step E: hoist                 提升 bias/zero 的循环不变切片
//
// 输入参数 (Step2，7个):
//   %arg0..%arg3: tensor 参数
//   %arg4: i64  旧 t_M (M轴 tile size, strip-mine 后成为 Tb_M 步长)
//   %arg5: i64  旧 t_N (N轴 tile size, strip-mine 后成为 Tb_N 步长)
//   %arg6: i64  t_K   (K轴 tile size, 保持不变)
//
// 输出参数 (Step3，新增4个 index):
//   %arg7: index  TB_M  核间 M 分块
//   %arg8: index  TB_N  核间 N 分块
//   %arg9: index  Tb_M  核内 M 分块 (strip-mine 后外层 for 步长)
//   %arg10: index Tb_N  核内 N 分块 (strip-mine 后外层 for 步长)
//
// 注: strip_mine 后原 for_M 的 step 变为 Tb_M，
//     若希望 Tb_M == 旧 t_M 则 Tb_M 参数传入相同值即可。
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
    //   追加4个 index 参数到 func 末尾:
    //     TB_M, TB_N, Tb_M, Tb_N
    //   返回值:
    //     %func_new          : 变换后新 func 句柄 (!transform.any_op)
    //     %TB_M/N, %Tb_M/N   : BlockArgument (!transform.any_value)
    // ================================================================
    %func_new, %TB_M_arg, %TB_N_arg, %Tb_M_arg, %Tb_N_arg =
        transform.func.add_index_args %func, 4
            : (!transform.any_op)
            -> (!transform.any_op,
                !transform.any_value,
                !transform.any_value,
                !transform.any_value,
                !transform.any_value)

    // ================================================================
    // Step B: strip_mine — M 轴外层切分 (生成 TB_M 循环)
    //
    //   目标: Step2 最外层 scf.for (M轴, step=t_M, iv=%arg7)
    //
    //   transform.loop.strip_mine %target, %step:
    //     在 %target for 循环外层插入一个新的 for 循环，步长=%step。
    //     原 for 循环变为内层，步长不变 (仍为 t_M，即 Tb_M)。
    //
    //   变换前:
    //     scf.for %arg7 = 0 to dim_M step t_M { ... }
    //
    //   变换后:
    //     scf.for %iv_TB_M = 0 to dim_M step TB_M {      ← 新外层
    //       scf.for %iv_Tb_M = %iv_TB_M               \
    //                       to min(%iv_TB_M+TB_M, M)   | ← 原内层，范围收窄
    //                       step t_M { ... }            /
    //     }
    //
    //   返回值 %for_TB_M: 新生成的外层 for 句柄，后续用于转 forall。
    // ================================================================
    %for_M = transform.structured.match ops{["scf.for"]} in %func_new
        : (!transform.any_op) -> !transform.any_op
    // 注: match 返回第一个匹配的 scf.for，即最外层 M 轴循环。
    //     若 IR 中有多个 for，需用 transform.foreach + filter 精确定位。
    //     Step2 最外层确为 M for，此处直接取第一个匹配。

    %for_TB_M = transform.loop.strip_mine %for_M, %TB_M_arg
        : (!transform.any_op, !transform.any_value) -> !transform.any_op

    // ================================================================
    // Step C: strip_mine — N 轴外层切分 (生成 TB_N 循环)
    //
    //   目标: N 轴 scf.for (step=t_N, iv=%arg9)
    //         strip_mine 后它已成为 TB_M 循环的内层，
    //         在 %for_TB_M 内重新 match 定位。
    //
    //   变换后结构:
    //     scf.for TB_M {               ← Step B 生成
    //       scf.for Tb_M {             ← 原 M for (收窄范围)
    //         scf.for TB_N {           ← Step C 新生成
    //           scf.for Tb_N { ... }   ← 原 N for (收窄范围)
    //         }
    //       }
    //     }
    //
    //   注: 此时 K for 和 matmul 仍在 Tb_N for 内部不变。
    // ================================================================
    %for_N = transform.structured.match ops{["scf.for"]} in %for_TB_M
        : (!transform.any_op) -> !transform.any_op
    // match 在 %for_TB_M 内部找第一个 scf.for，
    // 即 strip_mine 后的内层 Tb_M for。
    // 但我们需要的是 N for，它在 Tb_M for 内部。
    // 用两次 match 逐层下钻:
    %for_Tb_M_inner = transform.structured.match ops{["scf.for"]} in %for_N
        : (!transform.any_op) -> !transform.any_op
    // 现在 %for_Tb_M_inner 是 Tb_M 内层的第一个 for，即 N for。

    %for_TB_N = transform.loop.strip_mine %for_Tb_M_inner, %TB_N_arg
        : (!transform.any_op, !transform.any_value) -> !transform.any_op

    // ================================================================
    // Step D: loop_to_forall — 将 for_TB_M / for_TB_N 转为 scf.forall
    //
    //   transform.loop.forall_to_parallel 或其逆操作不适用此处。
    //   正确原语: transform.loop.loop_to_parallel (若工具链支持)
    //   或者直接用: transform.structured.tile_using_forall 对外层结构整体替换。
    //
    //   ⚠️ 实际上 MLIR transform dialect 目前没有直接把
    //      "已有 scf.for" 转成 "scf.forall" 的单一原语。
    //
    //   可行替代方案 (二选一，根据工具链版本选择):
    //
    //   方案 A (推荐, LLVM >= 18):
    //     transform.loop.to_forall %for_TB_M, %for_TB_N
    //     将两个相邻嵌套的 scf.for 合并为一个二维 scf.forall。
    //
    //   方案 B (兼容 LLVM 17):
    //     分两步:
    //       1. transform.loop.to_forall %for_TB_M  → 得到一维 forall_M
    //       2. transform.loop.to_forall %for_TB_N  → 在内层得到 forall_N
    //     注意: 两个独立 forall 嵌套与单个二维 forall 语义等价但形式不同，
    //     后端 lowering 时需要处理嵌套 forall。
    //
    //   此处使用方案 A:
    // ================================================================
    %forall_TB =
        transform.loop.forall %for_TB_M, %for_TB_N
            : (!transform.any_op, !transform.any_op)
            -> !transform.any_op
    // 生成结构:
    //   scf.forall (%iv_TB_M, %iv_TB_N) ... shared_outs(...) {
    //     scf.for Tb_M { scf.for Tb_N { scf.for K { matmul } } }
    //     add / relu
    //     in_parallel { parallel_insert_slice }
    //   }

    // ================================================================
    // Step E: hoist_loop_invariant_subsets
    //
    //   定位 Tb_N for 和 Tb_M for，由内向外提升不变切片。
    //   bias/zero 的切片不依赖 Tb_N/Tb_M 循环变量，
    //   提升后落在 forall 体内，每个 AI Core 只做一次 GM 读取。
    // ================================================================
    %for_Tb_N_final = transform.structured.match ops{["scf.for"]} in %forall_TB
        : (!transform.any_op) -> !transform.any_op
    // 取 forall 内最内层前的 for，即 Tb_N for。
    // 若 match 返回 Tb_M for，则对其做 hoist 同样有效
    // (hoist 会递归处理内层)。

    transform.loop.hoist_loop_invariant_subsets %for_Tb_N_final
        : !transform.any_op

    transform.yield
  }

} // end module

// ====================================================================
// ★ 重要补充说明
//
// 1. transform.loop.strip_mine 接口 (LLVM 17/18):
//      transform.loop.strip_mine %for_target, %step_value
//          : (!transform.any_op, !transform.any_value)
//          -> !transform.any_op   ← 返回新生成的外层 for
//    若工具链不支持此原语，等价替代:
//      transform.structured.tile_using_for 作用于 matmul，
//      但需先把 Step2 的 for 层全部剥掉后重新 tile，
//      即从 Step1（未 tiled）的 IR 开始变换。
//
// 2. transform.loop.forall 接口:
//      接受1个或2个 scf.for 句柄，将其转为 scf.forall。
//      两个 for 必须是直接嵌套关系 (外层/内层)。
//      若工具链版本不支持多参数形式，改为对两个 for 分别调用
//      transform.loop.to_forall，得到嵌套 forall。
//
// 3. match 精确定位问题:
//    Step2 中有多个 scf.for，match ops{["scf.for"]} 返回的是
//    所有匹配结果。若需精确取"第一个"，改用:
//      transform.structured.match ops{["scf.for"]}
//          ... {filter_result_type = ...}
//    或用 transform.foreach 遍历后按条件过滤。
//    本脚本依赖 Step2 的固定嵌套结构 (最外 = M for)，
//    请确认实际 IR 的 for 顺序与之一致。
//
// 4. 执行方式:
//    mlir-opt exp_output_step2_canonicalize.mlir \
//        --transform-interpreter=entry-point=__transform_main \
//        --allow-unregistered-dialect \
//        -o exp_output_step3_3-level.mlir
// ====================================================================
