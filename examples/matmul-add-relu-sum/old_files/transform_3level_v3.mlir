// transform_3level.mlir
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │ 报错根因                                                              │
// │                                                                      │
// │ tile_using_forall 的 tile_sizes 只接受静态整数字面量，                │
// │ 不接受任何动态 SSA value（无论 !transform.any_op 还是 any_param）。   │
// │                                                                      │
// │ 动态参数只能通过 num_threads 传入（指定核数，tile size 自动推导）。   │
// │                                                                      │
// │ 两种修复方案见下方，用 // [方案A] 和 // [方案B] 标注。              │
// └─────────────────────────────────────────────────────────────────────┘
//
// 函数新增入参 (3个 index，由 add_index_args 注入):
//   %sz#0 = Tb_M  核内 M 步长 (tile_using_for, Tb级)
//   %sz#1 = Tb_N  核内 N 步长 (tile_using_for, Tb级)
//   %sz#2 = t_K   CUBE K 步长 (tile_using_for, t级)
//
// 方案A: TB_M/TB_N 硬编码为静态 tile_sizes，适合 tile size 编译期确定的场景
// 方案B: 通过 num_threads 传入动态核数，tile size 由运行时矩阵大小/核数推导
//
// 执行:
//   mlir-opt fc_add_relu.mlir \
//     --transform-interpreter=entry-point=__transform_main \
//     --canonicalize --cse \
//     -o out_3level.mlir

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}
  ) {

    // ============================================================
    // Step 1: 匹配 func.func
    // ============================================================
    %func = transform.structured.match ops{["func.func"]}
              in %arg1
            : (!transform.any_op) -> !transform.any_op

    // ============================================================
    // Step 2: 为 func.func 增加 3 个 index 入参
    //   %sz#0 = Tb_M, %sz#1 = Tb_N, %sz#2 = t_K
    //   用于 tile_using_for，类型 !transform.any_op
    //
    // [方案B 额外增加2个参数]:
    //   若选方案B，改为 add_index_args %func, 5
    //   %sz#3 = num_cores_M (M方向核数)
    //   %sz#4 = num_cores_N (N方向核数)
    // ============================================================
    %transformed, %sz:3 = transform.func.add_index_args %func, 3
      : (!transform.any_op)
        -> (!transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op)

    // ============================================================
    // Step 3: 匹配 linalg ops
    // ============================================================
    %matmul = transform.structured.match ops{["linalg.matmul"]}
                in %transformed
              : (!transform.any_op) -> !transform.any_op

    %elementwise = transform.structured.match ops{["linalg.elementwise"]}
                     in %transformed
                   : (!transform.any_op) -> !transform.any_op

    %add, %max = transform.split_handle %elementwise
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 4 (继承原逻辑): 倒序 TileAndFuse，建立 Tb 级两层循环
    // ============================================================

    // tile max → Tb_M(%loop0) x Tb_N(%loop1) 两层 scf.for
    %tiled_max, %loop0, %loop1 =
        transform.structured.tile_using_for %max
            tile_sizes [%sz#0, %sz#1]
        : (!transform.any_op, !transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // fuse add 进 loop1 (Tb_N 内层)
    %add_fused, %loop2 =
        transform.structured.fuse_into_containing_op %add into %loop1
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // fuse matmul 进 loop1
    %matmul_fused, %loop3 =
        transform.structured.fuse_into_containing_op %matmul into %loop1
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // K 轴 tile step=t_K → t 级
    %matmul_fused_split:3 = transform.split_handle %matmul_fused
      : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %ret, %loop_K =
        transform.structured.tile_using_for %matmul_fused_split#0
            tile_sizes [0, 0, %sz#2]
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 5: TB 级并行化
    //
    // 重新 match max (之前的 handle 经 fuse 后已失效)
    // ============================================================
    %max_for_forall =
        transform.structured.match ops{["linalg.elementwise"]}
          attributes{kind = #linalg.elementwise_kind<max_signed>}
          in %transformed
        : (!transform.any_op) -> !transform.any_op

    // ----------------------------------------------------------
    // [方案A] 静态 tile_sizes (TB_M/TB_N 编译期硬编码)
    //
    //   tile_using_forall 只支持静态整数字面量作为 tile_sizes，
    //   动态 SSA value 在此不适用。
    //   适合场景: TB 大小在编译期固定（如昇腾910B单算子场景）。
    //
    //   修改 TB_M/TB_N 的值时直接改这两个数字即可。
    // ----------------------------------------------------------
    %tiled_forall, %forall_op =
        transform.structured.tile_using_forall %max_for_forall
            tile_sizes [256, 256]
            ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ----------------------------------------------------------
    // [方案B] 动态核数 num_threads (TB size 由运行时推导)
    //
    //   将上方 [方案A] 替换为下面的写法（二选一，不能同时启用）:
    //   需将 Step 2 的 add_index_args 改为 5 个参数，
    //   %sz#3=num_cores_M, %sz#4=num_cores_N。
    //
    //   num_threads 语义: 将迭代空间均分给指定数量的核，
    //   每核 tile size = ceil(dim / num_threads)，自动处理边界。
    //   适合场景: 核数固定、矩阵大小动态变化的场景。
    //
    //   %tiled_forall, %forall_op =
    //       transform.structured.tile_using_forall %max_for_forall
    //           num_threads [%sz#3, %sz#4]
    //           ( mapping = [#gpu.block<y>, #gpu.block<x>] )
    //       : (!transform.any_op, !transform.any_op, !transform.any_op)
    //         -> (!transform.any_op, !transform.any_op)
    // ----------------------------------------------------------

    // ============================================================
    // Step 6: 清理
    // ============================================================
    transform.apply_patterns to %transformed {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.apply_cse to %transformed : !transform.any_op

    transform.yield
  }
}
