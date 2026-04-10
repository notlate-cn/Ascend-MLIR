// transform_3level.mlir
//
// 在 transform_tile_and_fuse.mlir 基础上扩展，完成 TB/Tb/t 三级变换
//
// ┌─────────────────────────────────────────────────────────────────┐
// │ 报错根因与修复                                                    │
// │                                                                  │
// │ tile_using_forall 使用动态 size 时，每个动态 handle 的类型必须    │
// │ 紧跟在方括号内写出，格式:                                         │
// │   tile_sizes [%sz : !transform.any_param, ...]                   │
// │                                                                  │
// │ 而 add_index_args 返回的是 !transform.any_op，不是 any_param，    │
// │ 直接传给 tile_using_forall 会导致 operands/types 数量不匹配。     │
// │                                                                  │
// │ 修复: TB_M/TB_N 用 transform.param.constant 定义为 any_param，   │
// │       Tb_M/Tb_N/t_K 继续用 add_index_args 返回的 any_op。        │
// └─────────────────────────────────────────────────────────────────┘
//
// 函数新增入参 (3个 index，由 add_index_args 注入，类型 !transform.any_op):
//   %sz#0 = Tb_M : 核内 M 步长  → tile_using_for (Tb级)
//   %sz#1 = Tb_N : 核内 N 步长  → tile_using_for (Tb级)
//   %sz#2 = t_K  : CUBE K 步长  → tile_using_for (t级)
//
// TB_M / TB_N: transform.param.constant → !transform.any_param
//   → tile_using_forall (TB级)
//   如需运行时动态化，可改为从外部 named_sequence 参数注入 any_param。
//
// 变换后循环结构:
//   scf.forall (%iv_TB_M, %iv_TB_N) step(TB_M, TB_N)  <- TB: 核间并行
//     scf.for %iv_Tb_M step Tb_M                       <- Tb: 核内M循环
//       scf.for %iv_Tb_N step Tb_N                     <- Tb: 核内N循环
//         scf.for %iv_K   step t_K                     <- t:  K-reduction
//           linalg.matmul
//         linalg.elementwise add
//         linalg.elementwise max (relu)
//     in_parallel { tensor.parallel_insert_slice }
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
    //   类型: !transform.any_op  (tile_using_for 接受此类型)
    // ============================================================
    %transformed, %sz:3 = transform.func.add_index_args %func, 3
      : (!transform.any_op)
        -> (!transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op)

    // ============================================================
    // Step 3: TB 级 tile size —— param.constant → !transform.any_param
    //   tile_using_forall 要求动态 size 类型为 !transform.any_param
    //   与 add_index_args 返回的 !transform.any_op 是两种不同类型
    // ============================================================
    %TB_M = transform.param.constant 256 : i64 -> !transform.any_param
    %TB_N = transform.param.constant 256 : i64 -> !transform.any_param

    // ============================================================
    // Step 4: 匹配 linalg ops
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
    // Step 5 (继承原逻辑): 倒序 TileAndFuse，建立 Tb 级两层循环
    //
    //   先 tile max → Tb_M x Tb_N 两层 scf.for
    //   再倒序 fuse add、matmul 进内层 loop1 (Tb_N)
    //   最后 tile matmul K 轴 step=t_K
    //
    //   tile_using_for: size 类型 !transform.any_op，直接用 %sz#N
    // ============================================================
    %tiled_max, %loop0, %loop1 =
        transform.structured.tile_using_for %max
            tile_sizes [%sz#0, %sz#1]
        : (!transform.any_op, !transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %add_fused, %loop2 =
        transform.structured.fuse_into_containing_op %add into %loop1
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    %matmul_fused, %loop3 =
        transform.structured.fuse_into_containing_op %matmul into %loop1
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    %matmul_fused_split:3 = transform.split_handle %matmul_fused
      : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %ret, %loop_K =
        transform.structured.tile_using_for %matmul_fused_split#0
            tile_sizes [0, 0, %sz#2]
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 6 (新增): TB 级并行化 → scf.forall
    //
    //   对最终 consumer (max) 做 tile_using_forall
    //   将内部已 fuse 的 Tb 循环整体包进 forall 体内
    //
    //   关键: size 类型标注必须写在 [] 内紧跟每个动态 handle 之后
    //     tile_sizes [%TB_M : !transform.any_param,
    //                 %TB_N : !transform.any_param]
    //   这正是修复原报错的核心改动
    // ============================================================
    %max_for_forall =
        transform.structured.match ops{["linalg.elementwise"]}
          attributes{kind = #linalg.elementwise_kind<max_signed>}
          in %transformed
        : (!transform.any_op) -> !transform.any_op

    %tiled_forall, %forall_op =
        transform.structured.tile_using_forall %max_for_forall
            tile_sizes [%TB_M : !transform.any_param,
                        %TB_N : !transform.any_param]
            ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 7: 清理
    // ============================================================
    transform.apply_patterns to %transformed {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.apply_cse to %transformed : !transform.any_op

    transform.yield
  }
}
