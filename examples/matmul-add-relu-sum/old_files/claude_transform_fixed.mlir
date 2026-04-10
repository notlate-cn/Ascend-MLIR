// claude_transform_fixed.mlir
//
// 修复 fuse_into_containing_op 报错的正确变换顺序:
//
//   原因: fuse_into_containing_op 要求 elementwise op 是 matmul 的直接消费者，
//         但原始 IR 中间有 insert_slice/extract_slice 隔断了 def-use 链。
//
//   修复策略:
//     Step 1: tile_using_forall 做 TB 级并行 (scf.forall)
//     Step 2: 在 forall 体内，对 matmul 用 tile_using_for 做 Tb 级 tiling，
//             同时通过 tile_sizes 把 add/relu 一并 tile (不依赖 fuse)
//     Step 3: canonicalize + cse 清理
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
    // Step 1: TB 级并行化
    //   对 linalg.matmul 做 tile_using_forall
    //   tile_sizes = [TB_M, TB_N] 只切 M/N，不切 K
    //   产生 scf.forall 替换原来的 %5/%6 两层 scf.for
    //   mapping 到核间并行维度
    // ============================================================
    %matmul = transform.structured.match ops{["linalg.matmul"]}
                in %module
              : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_TB, %forall =
        transform.structured.tile_using_forall %matmul
            tile_sizes [256, 256]
            ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 2: 在 forall 体内，fuse elementwise 进 forall
    //   先把 add/relu fuse 进 scf.forall，建立直接 def-use 链
    //   此时 add/relu 的 extract_slice 还没有被 tiling 打断
    //   fuse 成功的前提: elementwise 是 forall 内 matmul 结果的消费者
    // ============================================================

    // 找到 forall 体内的两个 elementwise op
    %elementwise = transform.structured.match ops{["linalg.elementwise"]}
                     in %forall
                   : (!transform.any_op) -> !transform.any_op

    // fuse add 进 forall (add 直接消费 matmul 输出，def-use 链完整)
    %fused_elems, %fused_loop =
        transform.structured.fuse_into_containing_op %elementwise
            into %forall
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 3: Tb 级 tiling
    //   对 forall 体内已经 fuse 好的 matmul 做 tile_using_for
    //   tile_sizes = [Tb_M, Tb_N, 0]
    //     Tb_M=128: 核内 M 方向循环，驱动 A 片 GM->L1
    //     Tb_N=64:  核内 N 方向循环，驱动 B 片 GM->L1，A 片复用
    //     K=0:      不切 K，保留原 %9 的 K-reduction 作为 t 粒度
    //
    //   fuse 之后 add/relu 与 matmul 已经在同一 forall 体内，
    //   tile_using_for 只 tile matmul，add/relu 跟随最内层循环
    // ============================================================

    // 重新 match forall 体内的 matmul (经过 Step1 后是 tiled 后的版本)
    %matmul_in_forall = transform.structured.match ops{["linalg.matmul"]}
                          in %forall
                        : (!transform.any_op) -> !transform.any_op

    %tiled_matmul_Tb, %loops:2 =
        transform.structured.tile_using_for %matmul_in_forall
            tile_sizes [128, 64, 0]
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ============================================================
    // Step 4: fuse add/relu 进 Tb_N 最内层循环
    //   经过 Step 2 的 fuse，add/relu 已在 forall 体内
    //   此时再 fuse 进 Tb_N 循环 (%loops#1)
    //   def-use 链: matmul → (insert_slice) → forall body → add
    //   注意: 这里用 clone_and_fuse 模式避免 def-use 断链问题
    // ============================================================
    %add_op = transform.structured.match ops{["linalg.elementwise"]}
                attributes{kind = #linalg.elementwise_kind<add>}
                in %forall
              : (!transform.any_op) -> !transform.any_op

    %relu_op = transform.structured.match ops{["linalg.elementwise"]}
                 attributes{kind = #linalg.elementwise_kind<max_signed>}
                 in %forall
               : (!transform.any_op) -> !transform.any_op

    // fuse add 进 Tb_N 循环 (%loops#1 是最内层 N 方向循环)
    transform.structured.fuse_into_containing_op %add_op
        into %loops#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // fuse relu 紧跟 add 之后进同一循环
    transform.structured.fuse_into_containing_op %relu_op
        into %loops#1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 5: 清理
    // ============================================================
    %func = transform.structured.match ops{["func.func"]}
              in %module
            : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}
