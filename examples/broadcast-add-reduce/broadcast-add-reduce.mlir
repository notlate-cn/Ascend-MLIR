// ============================================================
//  AutoFuse 完整MLIR表示
//  计算图：A[M] → Broadcast → C[M,N]
//          C[M,N] + B[M,N] → D[M,N]   (Add)
//          D[M,N] → ReduceSum(axis=1) → E[M]
//
//  与 broadcast-add.mlir 相比，新增了 ReduceSum 这一 Reduce 类 Op。
//  这会导致轴分组不一致（d1变成Reduce轴），从而无法全局融合，
//  需要分段处理并生成2个Kernel模板。
//
//  编译流程：
//    STAGE 0  High-Level IR（tensor/linalg，完全符号化）
//    STAGE 1  轴分组分析（Parallel/Reduce/Others）
//    STAGE 2  融合判定与分段（Stage2a=可融合段，Stage2b=Reduce段）
//    STAGE 3  各段内部轴合并
//    STAGE 4  Tiling —— 每段对应一个Kernel模板
//    STAGE 5  TilingData 结构 + tiling_func（host侧符号化计算）
//    STAGE 6  Runtime Dispatch（性能模型选模板 + 多核下发）
//    STAGE 7  AscendC Lowering 目标代码伪表示
// ============================================================


// ============================================================
// STAGE 0: 输入的 High-Level IR（完全符号化，M/N 动态Shape）
// ============================================================
//
//   A : tensor<?xf16>      shape=[M]      Load1
//   B : tensor<?x?xf16>    shape=[M,N]    Load2
//   E : tensor<?xf16>      shape=[M]      输出（ReduceSum结果）
//
//   Op1: Broadcast  A[M] → C[M,N]
//   Op2: Add        D[M,N] = C[M,N] + B[M,N]
//   Op3: ReduceSum  E[M]  = sum_over_N( D[M,N] )

func.func @stage0_input(
    %A : tensor<?xf16>,       // shape [M]
    %B : tensor<?x?xf16>      // shape [M, N]
) -> tensor<?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %A, %c0 : tensor<?xf16>
  %N  = tensor.dim %B, %c1 : tensor<?x?xf16>

  // ── Op1: Broadcast ──────────────────────────────────────
  // A[M] → C[M,N]，沿 d1(N轴) 广播
  %empty_C = tensor.empty(%M, %N) : tensor<?x?xf16>
  %C = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A：只访问d0，d1被广播
      affine_map<(d0, d1) -> (d0, d1)>   // C：输出
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%A : tensor<?xf16>)
    outs(%empty_C : tensor<?x?xf16>) {
  ^bb0(%a_val: f16, %c_out: f16):
    linalg.yield %a_val : f16
  } -> tensor<?x?xf16>

  // ── Op2: Add ────────────────────────────────────────────
  // D[M,N] = C[M,N] + B[M,N]
  %empty_D = tensor.empty(%M, %N) : tensor<?x?xf16>
  %D = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // C
      affine_map<(d0, d1) -> (d0, d1)>,  // B
      affine_map<(d0, d1) -> (d0, d1)>   // D
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C, %B : tensor<?x?xf16>, tensor<?x?xf16>)
    outs(%empty_D : tensor<?x?xf16>) {
  ^bb0(%c_val: f16, %b_val: f16, %d_out: f16):
    %sum = arith.addf %c_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  // ── Op3: ReduceSum(axis=1) ──────────────────────────────
  // E[M] = sum_j D[M, j]，d1 是 reduce 轴
  %zero = arith.constant 0.0 : f16
  %empty_E = tensor.empty(%M) : tensor<?xf16>
  %init_E  = linalg.fill ins(%zero : f16)
               outs(%empty_E : tensor<?xf16>) -> tensor<?xf16>
  %E = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // D：输入
      affine_map<(d0, d1) -> (d0)>        // E：输出，d1被规约
    ],
    iterator_types = ["parallel", "reduction"]  // d0=Parallel, d1=Reduce
  } ins(%D : tensor<?x?xf16>)
    outs(%init_E : tensor<?xf16>) {
  ^bb0(%d_val: f16, %acc: f16):
    %new_acc = arith.addf %acc, %d_val : f16
    linalg.yield %new_acc : f16
  } -> tensor<?xf16>

  return %E : tensor<?xf16>
}


// ============================================================
// STAGE 1: 轴分组分析（编译器内部 Analysis，用注释展示）
//
//  对每个 Op 补齐到全局维度 [d0=M, d1=N]，分析 iterator_types：
//
//   Broadcast: iterator_types=["parallel","parallel"]
//     d0 → Parallel
//     d1 → Parallel（广播轴，输出侧parallel）
//
//   Add:       iterator_types=["parallel","parallel"]
//     d0 → Parallel
//     d1 → Parallel
//
//   ReduceSum: iterator_types=["parallel","reduction"]
//     d0 → Parallel
//     d1 → REDUCE  ← 与前两个Op的d1=Parallel 冲突！
//
//  全局轴分组冲突判定：
//    d0 : Broadcast=P, Add=P, Reduce=P  → 全局 Parallel ✅
//    d1 : Broadcast=P, Add=P, Reduce=R  → 类型不一致 ❌
//
//  → 无法全局融合！必须分段：
//    Segment A（Broadcast + Add）：d0=P, d1=P → 全Parallel，可融合
//    Segment B（ReduceSum）       ：d0=P, d1=R → Parallel+Reduce
//
//  → 生成 2 个独立 Kernel 模板：
//    Kernel_A: broadcast_add  输入A[M]+B[M,N] → 输出D[M,N]
//    Kernel_B: reduce_sum     输入D[M,N]      → 输出E[M]
// ============================================================


// ============================================================
// STAGE 2a: Segment A 融合后的 IR（Broadcast + Add → 单 generic）
// ============================================================

func.func @stage2a_fused_broadcast_add(
    %A : tensor<?xf16>,
    %B : tensor<?x?xf16>
) -> tensor<?x?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %A, %c0 : tensor<?xf16>
  %N  = tensor.dim %B, %c1 : tensor<?x?xf16>

  %empty_D = tensor.empty(%M, %N) : tensor<?x?xf16>
  %D = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A：广播
      affine_map<(d0, d1) -> (d0, d1)>,  // B
      affine_map<(d0, d1) -> (d0, d1)>   // D
    ],
    iterator_types = ["parallel", "parallel"],
    // 融合标记
    attrs = {fusion_group = 0 : i32, fusion_segment = "broadcast_add"}
  } ins(%A, %B : tensor<?xf16>, tensor<?x?xf16>)
    outs(%empty_D : tensor<?x?xf16>) {
  ^bb0(%a_val: f16, %b_val: f16, %out: f16):
    %sum = arith.addf %a_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  return %D : tensor<?x?xf16>
}


// ============================================================
// STAGE 2b: Segment B 的 IR（ReduceSum，无需融合变形）
// ============================================================

func.func @stage2b_reduce_sum(
    %D : tensor<?x?xf16>
) -> tensor<?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %D, %c0 : tensor<?x?xf16>

  %zero = arith.constant 0.0 : f16
  %empty_E = tensor.empty(%M) : tensor<?xf16>
  %init_E  = linalg.fill ins(%zero : f16)
               outs(%empty_E : tensor<?xf16>) -> tensor<?xf16>

  %E = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"],
    attrs = {fusion_group = 1 : i32, fusion_segment = "reduce_sum"}
  } ins(%D : tensor<?x?xf16>)
    outs(%init_E : tensor<?xf16>) {
  ^bb0(%d_val: f16, %acc: f16):
    %new_acc = arith.addf %acc, %d_val : f16
    linalg.yield %new_acc : f16
  } -> tensor<?xf16>

  return %E : tensor<?xf16>
}


// ============================================================
// STAGE 3a: Segment A 轴合并
//  [d0=M, d1=N] 均为 Parallel → 合并为 [d_flat=M*N]
//  与 broadcast-add.mlir STAGE 3 完全一致
// ============================================================

func.func @stage3a_axis_merged_broadcast_add(
    %A : tensor<?xf16>,
    %B : tensor<?x?xf16>
) -> tensor<?x?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %A, %c0 : tensor<?xf16>
  %N  = tensor.dim %B, %c1 : tensor<?x?xf16>
  %MN = arith.muli %M, %N : index

  %B_flat = tensor.collapse_shape %B [[0, 1]]
            : tensor<?x?xf16> into tensor<?xf16>

  %empty_flat = tensor.empty(%MN) : tensor<?xf16>
  %D_flat = linalg.generic {
    indexing_maps = [
      affine_map<(d_flat)[s0] -> (d_flat floordiv s0)>,  // A[row]，s0=N
      affine_map<(d_flat) -> (d_flat)>,                   // B_flat
      affine_map<(d_flat) -> (d_flat)>                    // D_flat
    ],
    iterator_types = ["parallel"],
    attrs = {axis_merged = true, segment = "broadcast_add",
             original_axes = "M,N", merged_symbol = "MN"}
  } ins(%A, %B_flat : tensor<?xf16>, tensor<?xf16>)
    outs(%empty_flat : tensor<?xf16>) {
  ^bb0(%a_val: f16, %b_val: f16, %out: f16):
    %sum = arith.addf %a_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?xf16>

  // 结果 reshape 回 [M,N]（供 Segment B 消费）
  %D_2d = tensor.expand_shape %D_flat [[0, 1]]
          output_shape [%M, %N]
          : tensor<?xf16> into tensor<?x?xf16>

  return %D_2d : tensor<?x?xf16>
}


// ============================================================
// STAGE 3b: Segment B 轴合并
//  ReduceSum：d0=Parallel(M)，d1=Reduce(N)
//  策略：Parallel轴在外层，Reduce轴在内层 → 不合并，保持2D结构
//        （两个不同类型的轴无法合并为同一类型）
//
//  注：若有多个连续Parallel轴或多个连续Reduce轴，可分别合并。
//  本例 M/N 各一个，保持原状，只标注轴角色供tiling使用。
// ============================================================

func.func @stage3b_reduce_sum_canonical(
    %D : tensor<?x?xf16>
) -> tensor<?xf16> {

  %c0 = arith.constant 0 : index
  %M  = tensor.dim %D, %c0 : tensor<?x?xf16>

  %zero = arith.constant 0.0 : f16
  %empty_E = tensor.empty(%M) : tensor<?xf16>
  %init_E  = linalg.fill ins(%zero : f16)
               outs(%empty_E : tensor<?xf16>) -> tensor<?xf16>

  // 保持 [Parallel=M, Reduce=N] 二维结构
  // 轴角色已内嵌在 iterator_types 中，tiling pass 据此生成 TB/Tb/t 结构
  %E = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"],
    attrs = {axis_canonical = true, segment = "reduce_sum",
             parallel_axes = "d0:M", reduce_axes = "d1:N"}
  } ins(%D : tensor<?x?xf16>)
    outs(%init_E : tensor<?xf16>) {
  ^bb0(%d_val: f16, %acc: f16):
    %new_acc = arith.addf %acc, %d_val : f16
    linalg.yield %new_acc : f16
  } -> tensor<?xf16>

  return %E : tensor<?xf16>
}


// ============================================================
// STAGE 4a: Tiling —— Kernel_A（broadcast+add，全Parallel）
//           与 broadcast-add.mlir STAGE 4 结构一致
//           Tiling结构：TB（核间）/ Tb（UB批次）/ t（向量化）
//           作用在 1D flat 空间
// ============================================================

// Transform Dialect 调度脚本（Kernel_A）
module attributes {transform.with_named_sequence} {
  transform.named_sequence @autofuse_tiling_KernelA(
      %root : !transform.any_op
  ) {
    %generic = transform.structured.match
        attributes {segment = "broadcast_add", axis_merged = true}
        in %root : (!transform.any_op) -> !transform.any_op

    // TB级：核间切分
    %TB_size = transform.param.constant 0 : i64  // 符号化，runtime填入
    %tiled_TB, %loop_TB = transform.structured.tile_using_for %generic [%TB_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    transform.loop.map_to_blocks %loop_TB {block_dims = [0]}
        : (!transform.any_op) -> ()

    // Tb级：UB搬运批次
    %Tb_size = transform.param.constant 0 : i64
    %tiled_Tb, %loop_Tb = transform.structured.tile_using_for %tiled_TB [%Tb_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    // t级：向量化粒度（128个f16）
    %t_size = transform.param.constant 128 : i64
    %tiled_t, %loop_t = transform.structured.tile_using_for %tiled_Tb [%t_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled_t : !transform.any_op

    transform.yield
  }
}

// Tiling后的 Affine Dialect IR（Kernel_A，1D flat 三级循环）
func.func @kernel_A_broadcast_add(
    %A      : memref<?xf16>,       // [M] GM
    %B_flat : memref<?xf16>,       // [M*N] GM（已collapse）
    %D_flat : memref<?xf16>,       // [M*N] GM 输出
    %MN      : index,
    %N       : index,              // 用于广播下标计算 row = flat_idx / N
    %TB_size : index,
    %Tb_size : index,
    %t_size  : index,
    %core_id : index
) {
  %c0 = arith.constant 0 : index

  // TB级：本核负责的起始偏移
  %tb_offset = arith.muli %core_id, %TB_size : index

  // Tb级循环：分批搬运+计算
  affine.for %tb = 0 to %TB_size step %Tb_size {
    %flat_start = arith.addi %tb_offset, %tb : index
    %remain     = arith.subi %TB_size, %tb : index
    %actual_Tb  = arith.minsi %Tb_size, %remain : index

    // 分配UB Buffer（on-chip SRAM）
    %buf_A   = memref.alloca(%Tb_size) : memref<?xf16, 9 : i32>   // VECIN
    %buf_B   = memref.alloca(%Tb_size) : memref<?xf16, 9 : i32>   // VECIN
    %buf_D   = memref.alloca(%Tb_size) : memref<?xf16, 10 : i32>  // VECOUT

    // DMA: B_flat 连续段 GM→UB
    memref.copy
        (memref.subview %B_flat[%flat_start][%actual_Tb][1]
         : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>),
        (memref.subview %buf_B[%c0][%actual_Tb][1]
         : memref<?xf16, 9 : i32> to memref<?xf16, strided<[1]>, 9 : i32>)
        : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>, 9 : i32>

    // Broadcast DMA: A GM→UB（按 flat_idx//N 取行号，gather语义）
    // 实现：逐元素 buf_A[i] = A[（flat_start+i）/ N]
    // 在真实lowering中映射到 AscendC GatherCopy 或标量循环
    affine.for %i = 0 to %actual_Tb step 1 {
      %abs_idx = arith.addi %flat_start, %i : index
      %row     = arith.divui %abs_idx, %N : index
      %a_elem  = memref.load %A[%row] : memref<?xf16>
      memref.store %a_elem, %buf_A[%i] : memref<?xf16, 9 : i32>
    }

    // t级：向量化 Add（128xf16 并行）
    affine.for %t = 0 to %actual_Tb step %t_size {
      %actual_t = arith.minsi %t_size, (arith.subi %actual_Tb, %t : index) : index
      %vec_a  = vector.load %buf_A[%t]  : memref<?xf16, 9 : i32>, vector<128xf16>
      %vec_b  = vector.load %buf_B[%t]  : memref<?xf16, 9 : i32>, vector<128xf16>
      %vec_d  = arith.addf %vec_a, %vec_b : vector<128xf16>
      vector.store %vec_d, %buf_D[%t] : memref<?xf16, 10 : i32>, vector<128xf16>
    }

    // DMA: D UB→GM 写回
    memref.copy
        (memref.subview %buf_D[%c0][%actual_Tb][1]
         : memref<?xf16, 10 : i32> to memref<?xf16, strided<[1]>, 10 : i32>),
        (memref.subview %D_flat[%flat_start][%actual_Tb][1]
         : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>)
        : memref<?xf16, strided<[1]>, 10 : i32> to memref<?xf16, strided<[1], offset: ?>>
  }

  return
}


// ============================================================
// STAGE 4b: Tiling —— Kernel_B（reducesum）
//
//  ReduceSum 结构：d0=Parallel(M)，d1=Reduce(N)
//
//  模板选择（2选1，runtime性能模型决定）：
//
//  Kernel_B_v1（Reduce-in-inner，适合 N 较小）：
//    for d0_TB (Parallel, 核间):
//      for d0_Tb (Parallel, UB批次):
//        DMA: D[row_start:row_end, :] → UB
//        for d0_t (向量化粒度):
//          for d1 (Reduce全N):        // 内层完整规约
//            UB_acc += UB_D[row, col]
//    TilingData_v1 = {TB_M, Tb_M, t_M, full_N}
//
//  Kernel_B_v2（Reduce-outer-split，适合 N 很大，UB放不下一行）：
//    for d1_outer_TB (Reduce分块, 核间):
//      for d0 (Parallel M):
//        for d1_inner_Tb (Reduce内层批次):
//          UB_acc += D[m, d1_inner:d1_inner+Tb_N]
//        // 需要二次 reduce（partial sum 写回GM再汇总）
//    TilingData_v2 = {TB_N, Tb_N, t_N, M}
// ============================================================

// Transform Dialect 调度脚本（Kernel_B_v1）
module attributes {transform.with_named_sequence} {
  transform.named_sequence @autofuse_tiling_KernelB_v1(
      %root : !transform.any_op
  ) {
    %generic = transform.structured.match
        attributes {segment = "reduce_sum", axis_canonical = true}
        in %root : (!transform.any_op) -> !transform.any_op

    // TB级：沿 Parallel 轴(d0=M) 核间切分
    %TB_M = transform.param.constant 0 : i64
    %tiled_TB, %loop_TB = transform.structured.tile_using_for %generic [%TB_M, 0]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    transform.loop.map_to_blocks %loop_TB {block_dims = [0]}
        : (!transform.any_op) -> ()

    // Tb级：沿 Parallel 轴(d0) 片内批次切分
    %Tb_M = transform.param.constant 0 : i64
    %tiled_Tb, %loop_Tb = transform.structured.tile_using_for %tiled_TB [%Tb_M, 0]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    // t级：沿 Parallel 轴向量化（Reduce轴不做向量化tile）
    %t_M = transform.param.constant 1 : i64  // 每次处理1行（含完整N规约）
    %tiled_t, %loop_t = transform.structured.tile_using_for %tiled_Tb [%t_M, 0]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled_t : !transform.any_op

    transform.yield
  }
}

// Tiling后的 Affine Dialect IR（Kernel_B_v1，M维三级 + N内层规约）
func.func @kernel_B_v1_reduce_sum(
    %D    : memref<?x?xf16>,  // [M, N] GM 输入
    %E    : memref<?xf16>,    // [M]    GM 输出
    %M    : index,
    %N    : index,
    %TB_M : index,            // Parallel轴TB切分量（per-core row数）
    %Tb_M : index,            // Parallel轴Tb切分量
    %t_M  : index,            // 向量化行数（通常=1，但可以并行多行）
    %core_id : index
) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %f0  = arith.constant 0.0 : f16

  // TB级：本核负责的行范围
  %row_start = arith.muli %core_id, %TB_M : index
  %row_end   = arith.addi %row_start, %TB_M : index
  // 边界裁剪
  %actual_end = arith.minsi %row_end, %M : index

  // Tb级循环：分批处理行
  affine.for %tb = %row_start to %actual_end step %Tb_M {
    %remain_rows = arith.subi %actual_end, %tb : index
    %actual_Tb   = arith.minsi %Tb_M, %remain_rows : index

    // 分配UB Buffer：D_tile[Tb_M, N] + acc[Tb_M]
    %buf_D   = memref.alloca(%Tb_M, %N) : memref<?x?xf16, 9 : i32>   // VECIN
    %buf_acc = memref.alloca(%Tb_M)     : memref<?xf16, 10 : i32>     // VECOUT
    linalg.fill ins(%f0 : f16) outs(%buf_acc : memref<?xf16, 10 : i32>)

    // DMA: D[tb:tb+actual_Tb, 0:N] GM→UB（二维tile搬运）
    memref.copy
        (memref.subview %D[%tb, %c0][%actual_Tb, %N][1, 1]
         : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>),
        (memref.subview %buf_D[%c0, %c0][%actual_Tb, %N][1, 1]
         : memref<?x?xf16, 9 : i32> to memref<?x?xf16, strided<[?, 1]>, 9 : i32>)
        : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1]>, 9 : i32>

    // t级（行粒度）+ Reduce轴内层循环
    affine.for %row = 0 to %actual_Tb step %t_M {
      // 对每行做完整的 N 规约（向量规约）
      // 对应 AscendC: ReduceSum(buf_acc[row], buf_D[row], N)
      affine.for %col = 0 to %N step 128 {
        %actual_n = arith.minsi (arith.constant 128 : index),
                                (arith.subi %N, %col : index) : index
        %vec_d   = vector.load %buf_D[%row, %col]
                   : memref<?x?xf16, 9 : i32>, vector<128xf16>
        %vec_acc = vector.load %buf_acc[%row]
                   : memref<?xf16, 10 : i32>, vector<1xf16>
        // 水平规约：sum of 128 elements
        %partial  = vector.reduction <add>, %vec_d : vector<128xf16> into f16
        %old_acc  = vector.extract %vec_acc[0] : f16 from vector<1xf16>
        %new_acc  = arith.addf %old_acc, %partial : f16
        %new_vec  = vector.splat %new_acc : vector<1xf16>
        vector.store %new_vec, %buf_acc[%row]
                     : memref<?xf16, 10 : i32>, vector<1xf16>
      }
    }

    // DMA: acc[Tb_M] UB→GM 写回 E[tb:tb+actual_Tb]
    memref.copy
        (memref.subview %buf_acc[%c0][%actual_Tb][1]
         : memref<?xf16, 10 : i32> to memref<?xf16, strided<[1]>, 10 : i32>),
        (memref.subview %E[%tb][%actual_Tb][1]
         : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>)
        : memref<?xf16, strided<[1]>, 10 : i32> to memref<?xf16, strided<[1], offset: ?>>
  }

  return
}


// ============================================================
// STAGE 5: TilingData 结构 + tiling_func（host侧，符号化计算）
//
//  两个Kernel各自的 TilingData：
//    TilingData_A：广播+加法（1D flat，同 broadcast-add.mlir）
//    TilingData_B_v1：规约（2D M×N，Reduce-in-inner 模板）
//    TilingData_B_v2：规约（2D M×N，Reduce-outer-split 模板）
// ============================================================

// Kernel_A TilingData（与 broadcast-add.mlir 相同）
!TilingData_A = !llvm.struct<"TilingData_A", (
    i64,  // M
    i64,  // N
    i64,  // MN = M*N
    i64,  // TB_size  (per-core flat 元素数)
    i64,  // Tb_size  (UB批次元素数)
    i64,  // t_size   (向量化宽度=128)
    i64   // core_num_A
)>

// Kernel_B_v1 TilingData（Reduce-in-inner，N不做切分）
!TilingData_B_v1 = !llvm.struct<"TilingData_B_v1", (
    i64,  // M
    i64,  // N         (完整 N，不切分)
    i64,  // TB_M      (per-core 行数)
    i64,  // Tb_M      (UB行批次)
    i64,  // t_M       (行向量化粒度，通常=1)
    i64   // core_num_B
)>

// Kernel_B_v2 TilingData（Reduce-outer-split，N分块）
!TilingData_B_v2 = !llvm.struct<"TilingData_B_v2", (
    i64,  // M
    i64,  // N
    i64,  // TB_N      (per-core N切块大小)
    i64,  // Tb_N      (UB N批次)
    i64,  // t_N       (向量化宽度=128)
    i64   // core_num_B
)>

func.func @tiling_func_A(%M : i64, %N : i64) -> !TilingData_A {
  // 硬件常量
  %UB_BYTES   = arith.constant 262144 : i64  // 256KB
  %CORE_NUM   = arith.constant 20 : i64
  %ELEM_BYTES = arith.constant 2 : i64       // f16=2B
  %VEC_WIDTH  = arith.constant 128 : i64

  %MN    = arith.muli %M, %N : i64
  %t_size = arith.constant 128 : i64

  // Tb_size：UB三等分（buf_A + buf_B + buf_D），对齐128
  %total_elems = arith.divsi %UB_BYTES, %ELEM_BYTES : i64
  %buf_elems   = arith.divsi %total_elems,
                              (arith.constant 3 : i64) : i64
  %Tb_size     = arith.muli
                   (arith.divsi %buf_elems, %t_size : i64),
                   %t_size : i64

  // TB_size：ceil(MN / CORE_NUM)，再对齐Tb
  %c1        = arith.constant 1 : i64
  %per_core  = arith.addi (arith.divsi %MN, %CORE_NUM : i64), %c1 : i64
  %TB_size   = arith.muli
                 (arith.addi
                   (arith.divsi %per_core, %Tb_size : i64), %c1 : i64),
                 %Tb_size : i64

  %core_num  = arith.addi
                 (arith.divsi %MN, %TB_size : i64), %c1 : i64

  %td = llvm.mlir.undef : !TilingData_A
  %td1 = llvm.insertvalue %M,        %td[0]  : !TilingData_A
  %td2 = llvm.insertvalue %N,        %td1[1] : !TilingData_A
  %td3 = llvm.insertvalue %MN,       %td2[2] : !TilingData_A
  %td4 = llvm.insertvalue %TB_size,  %td3[3] : !TilingData_A
  %td5 = llvm.insertvalue %Tb_size,  %td4[4] : !TilingData_A
  %td6 = llvm.insertvalue %t_size,   %td5[5] : !TilingData_A
  %td7 = llvm.insertvalue %core_num, %td6[6] : !TilingData_A
  return %td7 : !TilingData_A
}

func.func @tiling_func_B(%M : i64, %N : i64) -> !TilingData_B_v1 {
  // 对 v1（Reduce-in-inner）求解：前提是 N*2B <= UB/2
  // buf_D [Tb_M, N] + buf_acc [Tb_M] 需 <= UB
  %UB_BYTES   = arith.constant 262144 : i64
  %CORE_NUM   = arith.constant 20 : i64
  %ELEM_BYTES = arith.constant 2 : i64
  %c1         = arith.constant 1 : i64

  // UB中 buf_D 最多放多少行：floor(UB_BYTES / (N * 2))
  %row_bytes  = arith.muli %N, %ELEM_BYTES : i64
  %max_rows   = arith.divsi %UB_BYTES, %row_bytes : i64

  %t_M        = arith.constant 1 : i64  // 每次处理1行

  // Tb_M：对齐 t_M，不超过 max_rows
  %Tb_M       = arith.muli
                  (arith.divsi %max_rows, %t_M : i64),
                  %t_M : i64

  // TB_M：ceil(M / CORE_NUM)，对齐 Tb_M
  %per_core   = arith.addi (arith.divsi %M, %CORE_NUM : i64), %c1 : i64
  %TB_M       = arith.muli
                  (arith.addi
                    (arith.divsi %per_core, %Tb_M : i64), %c1 : i64),
                  %Tb_M : i64

  %core_num   = arith.addi
                  (arith.divsi %M, %TB_M : i64), %c1 : i64

  %td = llvm.mlir.undef : !TilingData_B_v1
  %td1 = llvm.insertvalue %M,        %td[0]  : !TilingData_B_v1
  %td2 = llvm.insertvalue %N,        %td1[1] : !TilingData_B_v1
  %td3 = llvm.insertvalue %TB_M,     %td2[2] : !TilingData_B_v1
  %td4 = llvm.insertvalue %Tb_M,     %td3[3] : !TilingData_B_v1
  %td5 = llvm.insertvalue %t_M,      %td4[4] : !TilingData_B_v1
  %td6 = llvm.insertvalue %core_num, %td5[5] : !TilingData_B_v1
  return %td6 : !TilingData_B_v1
}


// ============================================================
// STAGE 6: Runtime Dispatch
//
//  host侧逻辑：
//  1. 调用 tiling_func_A → TilingData_A
//  2. 调用 tiling_func_B → TilingData_B_v1
//  3. 性能模型判定：若 N * 2B <= UB/2（单行能放入UB）→ 选 Kernel_B_v1
//                   否则                              → 选 Kernel_B_v2
//  4. 先多核下发 Kernel_A（broadcast+add），生产 D[M,N]
//  5. 再多核下发 Kernel_B（reducesum），消费 D[M,N] → E[M]
//
//  注意：两个 Kernel 之间存在数据依赖，必须串行下发（同步屏障）。
//        同一 Kernel 内部的多核是并行的（scf.parallel）。
// ============================================================

func.func @runtime_dispatch(
    %A   : memref<?xf16>,    // [M]
    %B   : memref<?x?xf16>,  // [M, N]
    %E   : memref<?xf16>     // [M] 输出
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %M = memref.dim %A, %c0 : memref<?xf16>
  %N = memref.dim %B, %c1 : memref<?x?xf16>

  %M_i64 = arith.index_cast %M : index to i64
  %N_i64 = arith.index_cast %N : index to i64

  // ── Step 1: 计算两段的 TilingData ─────────────────────
  %td_A = func.call @tiling_func_A(%M_i64, %N_i64)
          : (i64, i64) -> !TilingData_A
  %td_B = func.call @tiling_func_B(%M_i64, %N_i64)
          : (i64, i64) -> !TilingData_B_v1

  // ── Step 2: 性能模型选 Kernel_B 模板 ──────────────────
  // 判断 N*2 <= UB_BYTES/2（v1 条件：单行能放入UB的一半）
  %UB_HALF    = arith.constant 131072 : i64  // 128KB
  %row_bytes  = arith.muli %N_i64, (arith.constant 2 : i64) : i64
  %use_v1     = arith.cmpi ule, %row_bytes, %UB_HALF : i64
  // （实际 dispatch 分支在此省略；演示时固定选 v1）

  // ── Step 3: 中间 Buffer D[M, N]（GM 上临时分配）───────
  %D_flat_buf = memref.alloc(%M, %N) {alignment = 64 : i64}
                : memref<?x?xf16>
  %D_flat     = memref.collapse_shape %D_flat_buf [[0, 1]]
                : memref<?x?xf16> into memref<?xf16>
  %B_flat     = memref.collapse_shape %B [[0, 1]]
                : memref<?x?xf16> into memref<?xf16>
  %MN         = arith.muli %M, %N : index

  // ── Step 4: 下发 Kernel_A（broadcast + add）───────────
  %core_A   = arith.index_cast
                (llvm.extractvalue %td_A[6] : !TilingData_A)
                : i64 to index
  %TB_A_idx = arith.index_cast
                (llvm.extractvalue %td_A[3] : !TilingData_A)
                : i64 to index
  %Tb_A_idx = arith.index_cast
                (llvm.extractvalue %td_A[4] : !TilingData_A)
                : i64 to index
  %t_A_idx  = arith.index_cast
                (llvm.extractvalue %td_A[5] : !TilingData_A)
                : i64 to index

  scf.parallel (%cid_A) = (%c0) to (%core_A) step (%c1) {
    func.call @kernel_A_broadcast_add(
        %A, %B_flat, %D_flat,
        %MN, %N,
        %TB_A_idx, %Tb_A_idx, %t_A_idx,
        %cid_A
    ) : (memref<?xf16>, memref<?xf16>, memref<?xf16>,
         index, index, index, index, index, index) -> ()
    scf.reduce
  }
  // 同步屏障：Kernel_A 全部完成后才能执行 Kernel_B
  // （在 MLIR 中用 scf.parallel 的自然结束表示；真实下发用 rtEvent）

  // ── Step 5: 下发 Kernel_B_v1（reduce sum）─────────────
  %core_B   = arith.index_cast
                (llvm.extractvalue %td_B[5] : !TilingData_B_v1)
                : i64 to index
  %TB_B_idx = arith.index_cast
                (llvm.extractvalue %td_B[2] : !TilingData_B_v1)
                : i64 to index
  %Tb_B_idx = arith.index_cast
                (llvm.extractvalue %td_B[3] : !TilingData_B_v1)
                : i64 to index
  %t_B_idx  = arith.index_cast
                (llvm.extractvalue %td_B[4] : !TilingData_B_v1)
                : i64 to index

  scf.parallel (%cid_B) = (%c0) to (%core_B) step (%c1) {
    func.call @kernel_B_v1_reduce_sum(
        %D_flat_buf, %E,
        %M, %N,
        %TB_B_idx, %Tb_B_idx, %t_B_idx,
        %cid_B
    ) : (memref<?x?xf16>, memref<?xf16>,
         index, index, index, index, index, index) -> ()
    scf.reduce
  }

  memref.dealloc %D_flat_buf : memref<?x?xf16>
  return
}


// ============================================================
// STAGE 7: AscendC Lowering 目标代码伪表示
//
// ── Kernel_A（broadcast + add）───────────────────────────
//
//  __global__ __aicore__ void kernel_broadcast_add(
//      GM_ADDR A, GM_ADDR B, GM_ADDR D, TilingData_A* td
//  ) {
//      int cid = GetBlockIdx();
//      int64_t flat_start = cid * td->TB_size;
//
//      for (int64_t tb = 0; tb < td->TB_size; tb += td->Tb_size) {
//          int64_t abs = flat_start + tb;
//          int64_t len = min(td->Tb_size, td->TB_size - tb);
//
//          // GM→UB：B 连续搬运
//          DataCopy(buf_B, B + abs, len);
//
//          // GM→UB：A 广播（gather：每个 flat_idx 读 A[flat_idx/N]）
//          for (int i = 0; i < len; i++)
//              buf_A[i] = A[(abs + i) / td->N];
//
//          // 向量加法
//          for (int t = 0; t < len; t += 128)
//              Add(buf_D + t, buf_A + t, buf_B + t, 128);
//
//          // UB→GM：D 写回
//          DataCopy(D + abs, buf_D, len);
//      }
//  }
//
// ── Kernel_B_v1（reduce sum，N能放入UB）───────────────────
//
//  __global__ __aicore__ void kernel_reduce_sum_v1(
//      GM_ADDR D, GM_ADDR E, TilingData_B_v1* td
//  ) {
//      int cid = GetBlockIdx();
//      int64_t row_start = cid * td->TB_M;
//      int64_t row_end   = min(row_start + td->TB_M, td->M);
//
//      for (int64_t rb = row_start; rb < row_end; rb += td->Tb_M) {
//          int64_t rows = min(td->Tb_M, row_end - rb);
//
//          // GM→UB：D[rb:rb+rows, 0:N]
//          DataCopy2D(buf_D, D + rb * td->N, rows, td->N);
//          // 初始化累加器
//          Duplicate(buf_acc, 0.0f16, rows);
//
//          // 对每行做向量规约
//          for (int r = 0; r < rows; r++) {
//              for (int c = 0; c < td->N; c += 128)
//                  ReduceAdd(buf_acc + r, buf_D + r*N + c, 128);
//          }
//
//          // UB→GM：E[rb:rb+rows]
//          DataCopy(E + rb, buf_acc, rows);
//      }
//  }
//
// ── Kernel_B_v2（reduce sum，N超过UB，Reduce外层分块）─────
//
//  __global__ __aicore__ void kernel_reduce_sum_v2(
//      GM_ADDR D, GM_ADDR partial, GM_ADDR E, TilingData_B_v2* td
//  ) {
//      int cid = GetBlockIdx();
//      int64_t col_start = cid * td->TB_N;    // 核间沿N轴切分
//
//      for (int m = 0; m < td->M; m++) {       // 每行单独处理
//          float16 acc = 0;
//          for (int nb = col_start;
//               nb < col_start + td->TB_N;
//               nb += td->Tb_N) {
//              int len = min(td->Tb_N, col_start+td->TB_N - nb);
//              DataCopy(buf_D, D + m*td->N + nb, len);
//              for (int t = 0; t < len; t += 128)
//                  ReduceAdd(&acc, buf_D + t, 128);
//          }
//          partial[cid * td->M + m] = acc;  // partial sum写回GM
//      }
//      // 第二轮：在 core 0 上对 partial[*][m] 求和写入 E[m]
//      // （或用 AtomicAdd，取决于硬件支持）
//  }
// ============================================================
