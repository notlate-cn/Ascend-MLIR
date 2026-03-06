// claude_transform_param.mlir
//
// claude_transform_fixed.mlir 的非硬编码版本
//
// 硬编码问题及解法:
//
//   原硬编码          解法
//   ─────────────────────────────────────────────────────────────
//   tile_sizes[256,256]  tile_using_forall 的 tile_sizes 只接受静态整数字面量，
//   (TB级, forall)       无法传入动态 SSA value。
//                        → 改用 num_threads 接口，传入"核数"而非"tile size"，
//                          num_threads 接受动态 !transform.any_op handle。
//                          tile size 由运行时推导: tile_sz = ceil(dim / num_cores)
//
//   tile_sizes[128,64,0] tile_using_for 支持动态 !transform.any_op handle
//   (Tb级, for)          → 直接用 add_index_args 注入，无需改接口
//
// 函数新增入参 (5个 index，由 add_index_args 注入):
//   arg[N+0] = num_cores_M : TB级 M方向核数 → tile_using_forall num_threads
//   arg[N+1] = num_cores_N : TB级 N方向核数 → tile_using_forall num_threads
//   arg[N+2] = Tb_M        : Tb级 M步长    → tile_using_for tile_sizes
//   arg[N+3] = Tb_N        : Tb级 N步长    → tile_using_for tile_sizes
//   arg[N+4] = t_K         : t级  K步长    → tile_using_for tile_sizes (K轴)
//
// 注意: 输入是 claude.mlir (已有 %5/%6 两层 scf.for 的版本)
//       若输入是 fc_add_relu.mlir (无循环的原始版本) 请用 transform_3level_v3.mlir
//
// 执行:
//   mlir-opt claude.mlir \
//     --transform-interpreter=entry-point=__transform_main \
//     --canonicalize --cse \
//     -o out.mlir

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.readonly}
  ) {

    // ============================================================
    // Step 0: 匹配 func.func，注入5个动态 index 参数
    // ============================================================
    %func = transform.structured.match ops{["func.func"]}
              in %module
            : (!transform.any_op) -> !transform.any_op

    %func_with_args, %sz:5 = transform.func.add_index_args %func, 5
      : (!transform.any_op)
        -> (!transform.any_op,
            !transform.any_op,   // %sz#0 = num_cores_M
            !transform.any_op,   // %sz#1 = num_cores_N
            !transform.any_op,   // %sz#2 = Tb_M
            !transform.any_op,   // %sz#3 = Tb_N
            !transform.any_op)   // %sz#4 = t_K

    // ============================================================
    // Step 1: TB 级并行化 → scf.forall
    //
    //   用 num_threads 代替 tile_sizes，接受动态 handle
    //   num_threads 语义: 将 [0, dim_M) x [0, dim_N) 均分给指定核数
    //   每核实际 tile size = ceil(dim / num_cores)，自动处理尾块
    //
    //   类型写法: num_threads 的动态 size 类型同样是 !transform.any_op
    //   : (!transform.any_op, !transform.any_op, !transform.any_op) -> (...)
    //     ^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
    //     目标 op                 num_cores_M     num_cores_N
    // ============================================================
    %matmul = transform.structured.match ops{["linalg.matmul"]}
                in %func_with_args
              : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_TB, %forall =
        transform.structured.tile_using_forall %matmul
            num_threads [%sz#0, %sz#1]
            ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op, !transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 2: fuse elementwise 进 forall
    //   在 Tb 级 tiling 之前先 fuse，保证 def-use 链完整
    //   (原 claude_transform_fixed 的修复策略，保持不变)
    // ============================================================
    %elementwise = transform.structured.match ops{["linalg.elementwise"]}
                     in %forall
                   : (!transform.any_op) -> !transform.any_op

    %fused_elems, %fused_loop =
        transform.structured.fuse_into_containing_op %elementwise
            into %forall
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 3: Tb 级 tiling → Tb_M x Tb_N 两层 scf.for
    //
    //   tile_using_for 支持动态 handle，直接用 %sz#2/%sz#3
    //   K 轴传 0 表示不切（保留原 K-reduction 作为 t 粒度）
    //
    //   类型写法: 每个动态 size 在末尾类型列表中对应一个 !transform.any_op
    //   : (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    //     ^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^     ^^^^^^^^^^^^
    //     目标 op                 Tb_M            Tb_N              (K=0 是静态，不占位)
    //
    //   注意: tile_sizes 列表中静态 0 不产生对应的类型占位，
    //         只有动态 handle (%sz#N) 需要在类型列表中有对应项
    // ============================================================
    %matmul_in_forall = transform.structured.match ops{["linalg.matmul"]}
                          in %forall
                        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb, %loops:2 =
        transform.structured.tile_using_for %matmul_in_forall
            tile_sizes [%sz#2, %sz#3, 0]
        : (!transform.any_op, !transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ============================================================
    // Step 4: fuse add/relu 进 Tb_N 最内层循环 (%loops#1)
    // ============================================================
    %add_op = transform.structured.match ops{["linalg.elementwise"]}
                attributes{kind = #linalg.elementwise_kind<add>}
                in %forall
              : (!transform.any_op) -> !transform.any_op

    %relu_op = transform.structured.match ops{["linalg.elementwise"]}
                 attributes{kind = #linalg.elementwise_kind<max_signed>}
                 in %forall
               : (!transform.any_op) -> !transform.any_op

    transform.structured.fuse_into_containing_op %add_op into %loops#1
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    transform.structured.fuse_into_containing_op %relu_op into %loops#1
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 5: t 级 K 轴 tiling
    //   对 Tb 层内的 matmul 再切 K 轴，step = t_K
    //   产生第三层 scf.for，对应 CUBE 单次 K-reduction 粒度
    // ============================================================
    %matmul_in_tb = transform.structured.match ops{["linalg.matmul"]}
                      in %forall
                    : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_t, %loop_K =
        transform.structured.tile_using_for %matmul_in_tb
            tile_sizes [0, 0, %sz#4]
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 6: 清理
    // ============================================================
    transform.apply_patterns to %func_with_args {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.apply_cse to %func_with_args : !transform.any_op

    transform.yield
  }
}
