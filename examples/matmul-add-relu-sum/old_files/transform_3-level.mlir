// 使用 Transform dialect 将 claude.mlir 变换为 TB/Tb/t 三级循环结构
//
// 执行方式:
//   mlir-opt claude.mlir \
//     --transform-interpreter=entry-point=__transform_main \
//     --canonicalize \
//     --cse \
//     -o claude_tiled_ascend.mlir
//
// Transform dialect 做的三件事:
//   Step 1: tile linalg.matmul → 产生 Tb_M x Tb_N 两层 scf.for (核内循环)
//           同时 extract_slice 的 offset/size 自动更新
//   Step 2: fuse elementwise (add/relu) 进 matmul 的 tiling 循环
//           保证 add/relu 留在 Tb_N 内，不上提
//   Step 3: 对外两层 scf.for (%5/%6, TB_M x TB_N) 做 forall 并行化

module attributes {transform.with_named_sequence} {

  // ----------------------------------------------------------------
  // 主变换序列, mlir-opt 以此为入口
  // ----------------------------------------------------------------
  transform.named_sequence @__transform_main(
      %module: !transform.any_op {transform.readonly}
  ) {

    // ============================================================
    // Step 0: 找到目标 op
    // ============================================================

    // 找到 func @fc_relu
    %func = transform.structured.match ops{["func.func"]}
              attributes{sym_name = "fc_relu"}
              in %module
            : (!transform.any_op) -> !transform.any_op

    // 找到 linalg.matmul (原始 %9 内部的那个)
    %matmul = transform.structured.match ops{["linalg.matmul"]}
                in %func
              : (!transform.any_op) -> !transform.any_op

    // 找到两个 linalg.elementwise (add 和 relu/max)
    %elementwise_ops = transform.structured.match ops{["linalg.elementwise"]}
                         in %func
                       : (!transform.any_op) -> !transform.any_op

    // ============================================================
    // Step 1: Tile linalg.matmul → Tb_M x Tb_N 两层核内 scf.for
    //
    // tile_sizes = [Tb_M, Tb_N, 0]
    //   - 前两维 (M, N) 分别按 Tb_M=128, Tb_N=64 切分 → 产生两层 scf.for
    //   - 第三维 (K) 置 0 表示不切 → 对应原 %9 的 K-reduction 保持不动
    //
    // 变换前 (在原 %5/%6 内部):
    //   linalg.matmul [M_tb, K] x [K, N_tb]
    //
    // 变换后:
    //   scf.for %iv_Tb_M = 0 to M_tb step 128    ← Tb_M 层
    //     scf.for %iv_Tb_N = 0 to N_tb step 64   ← Tb_N 层
    //       linalg.matmul [128, K] x [K, 64]      ← t 粒度 matmul
    //
    // extract_slice 的 offset/size 由 tiling 自动生成，无需手动调整
    // ============================================================
    %tiled_matmul, %loops:2 = transform.structured.tile_using_for %matmul
        tile_sizes [128, 64, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ============================================================
    // Step 2: Fuse elementwise ops 进 Tb_N 循环
    //
    // 将 add 和 relu fuse 进 matmul tiling 产生的最内层循环 (%loops#1 = Tb_N 层)
    // 保证 add/relu 在 Tb_N 内执行，不上提到 Tb_M 层
    //
    // fuse 做的事:
    //   - 把 elementwise op 的 extract_slice 移入循环体
    //   - 在循环体内紧接 matmul 之后执行 add/relu
    //   - 消除原来循环外的 elementwise op
    // ============================================================

    // fuse add 进 Tb_N 循环 (最内层 = %loops#1)
    %fused_add, %_ = transform.structured.fuse_into_containing_op
        %elementwise_ops
        into %loops#1
      : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 3: 将外两层 scf.for (%5/%6, TB_M x TB_N) 并行化为 scf.forall
    //
    // 找到原始的两层外循环 (tile_using_for 不会动它们，它们是 %5/%6)
    // 用 loop_to_forall 将串行 for 转为并行 forall
    //
    // 变换前:
    //   scf.for %iv_TB_M = 0 to dim_M step TB_M    ← %5
    //     scf.for %iv_TB_N = 0 to dim_N step TB_N  ← %6
    //
    // 变换后:
    //   scf.forall (%iv_TB_M, %iv_TB_N)
    //       in (%dim_M, %dim_N) step (%TB_M, %TB_N)
    //     ...
    //   in_parallel { tensor.parallel_insert_slice }
    // ============================================================

    // 找到变换后 IR 中的外层两个 scf.for (即原 %5/%6)
    // 注意: tile_using_for 在外层 for 内部插入了 Tb 层，外层本身未动
    %outer_loops = transform.structured.match ops{["scf.for"]}
                     in %func
                   : (!transform.any_op) -> !transform.any_op

    // 将最外层 scf.for 转为 scf.forall
    // mapping 指定并行维度到硬件资源 (这里用 #gpu.block 语义表示核间并行)
    transform.loop.forall_to_for %outer_loops
      : (!transform.any_op) -> ()

    // 实际并行化: 用 scf.forall 替换外两层 for
    // 找到 matmul 的父级循环链，取最外两层做 parallel tiling
    %parallel_tiled, %forall =
        transform.structured.tile_using_forall %matmul
            tile_sizes [256, 256]  // TB_M=256, TB_N=256
            // mapping 到 AI Core (用 #gpu.block 表达核间并行语义)
            ( mapping = [#gpu.block<y>, #gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // ============================================================
    // Step 4: 清理
    // ============================================================
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }
}
