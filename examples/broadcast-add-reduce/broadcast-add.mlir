// ============================================================
//  AutoFuse 完整MLIR表示
//  计算图：Load1[M] + Load2[M,N] + Broadcast + Add + Store[M,N]
//  编译流程：High-Level → 融合 → Tiling → Ascend后端
// ============================================================


// ============================================================
// STAGE 0: 输入的 High-Level IR（Linalg + Tensor方言）
//          完全符号化，M/N是动态Shape
// ============================================================

// 符号化Shape：?代表动态维度
// A: tensor<?xf16>      对应 shape=[M]
// B: tensor<?x?xf16>    对应 shape=[M,N]
// Out: tensor<?x?xf16>  对应 shape=[M,N]

func.func @entry_before_fusion(
    %A   : tensor<?xf16>,       // Load1 输入
    %B   : tensor<?x?xf16>     // Load2 输入
) -> tensor<?x?xf16> {

  // 获取符号化维度
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %A, %c0 : tensor<?xf16>       // 符号 M
  %N  = tensor.dim %B, %c1 : tensor<?x?xf16>     // 符号 N

  // ── Op1: Broadcast ──────────────────────────────────────
  // 语义：A[M] → C[M,N]，沿 d1(N轴) 广播
  // indexing_map: (d0,d1) -> (d0)  表示输出[d0,d1]读A[d0]
  %empty_C = tensor.empty(%M, %N) : tensor<?x?xf16>
  %C = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A 的访问：只用d0，d1被广播
      affine_map<(d0, d1) -> (d0, d1)>   // C 的访问：d0,d1都用
    ],
    iterator_types = ["parallel", "parallel"]  // d0=Parallel, d1=Parallel
  } ins(%A : tensor<?xf16>)
    outs(%empty_C : tensor<?x?xf16>) {
  ^bb0(%a_val: f16, %c_out: f16):
    linalg.yield %a_val : f16             // 直接复制，实现广播
  } -> tensor<?x?xf16>

  // ── Op2: Add ────────────────────────────────────────────
  // 语义：D[M,N] = C[M,N] + B[M,N]，纯Elementwise
  %empty_D = tensor.empty(%M, %N) : tensor<?x?xf16>
  %D = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // C 的访问
      affine_map<(d0, d1) -> (d0, d1)>,  // B 的访问
      affine_map<(d0, d1) -> (d0, d1)>   // D 的访问（输出）
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C, %B : tensor<?x?xf16>, tensor<?x?xf16>)
    outs(%empty_D : tensor<?x?xf16>) {
  ^bb0(%c_val: f16, %b_val: f16, %d_out: f16):
    %sum = arith.addf %c_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  // ── Store（函数返回即表示写出）──────────────────────────
  return %D : tensor<?x?xf16>
}


// ============================================================
// STAGE 1: 轴分组分析（编译器内部 Analysis，用注释表示）
//
// 对每个 linalg.generic 分析 indexing_maps：
//
//   Broadcast: iterator_types=["parallel","parallel"]
//     d0 → A访问(d0)   → Parallel（输入输出都覆盖）
//     d1 → A不访问d1   → 广播轴，仍归 Parallel（输出侧是parallel）
//
//   Add:       iterator_types=["parallel","parallel"]
//     d0 → Parallel
//     d1 → Parallel
//
//   融合判定：
//     全局 d0 = Parallel ✅
//     全局 d1 = Parallel ✅
//     → 所有轴分组一致 → 可融合！
// ============================================================


// ============================================================
// STAGE 2: 融合后的 IR（Broadcast + Add 合并为单个 generic）
//          通过 linalg fusion pass 生成
// ============================================================

func.func @entry_after_fusion(
    %A : tensor<?xf16>,
    %B : tensor<?x?xf16>
) -> tensor<?x?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %A, %c0 : tensor<?xf16>
  %N  = tensor.dim %B, %c1 : tensor<?x?xf16>

  // ── 融合后：Broadcast + Add → 单个 linalg.generic ────────
  // 两个Op合并，消除中间tensor C，直接在body中内联广播逻辑
  %empty_out = tensor.empty(%M, %N) : tensor<?x?xf16>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A：d1被广播，只访问d0
      affine_map<(d0, d1) -> (d0, d1)>,  // B：正常二维访问
      affine_map<(d0, d1) -> (d0, d1)>   // Out：输出
    ],
    iterator_types = ["parallel", "parallel"],  // 全局轴分组：[P, P]
    // 融合标记（自定义attribute）
    attrs = {fusion_group = 0 : i32}
  } ins(%A, %B : tensor<?xf16>, tensor<?x?xf16>)
    outs(%empty_out : tensor<?x?xf16>) {
  ^bb0(%a_val: f16, %b_val: f16, %out: f16):
    // 内联了 Broadcast 的语义（a_val已经是A[d0]，天然广播）
    %sum = arith.addf %a_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  return %result : tensor<?x?xf16>
}


// ============================================================
// STAGE 3: 轴合并（Axis Merging）
//          [d0=M, d1=N] 两个Parallel轴 → 合并为 [d_flat=M*N]
//          通过 linalg.reshape / collapse_shape 表达
// ============================================================

func.func @entry_after_axis_merge(
    %A : tensor<?xf16>,
    %B : tensor<?x?xf16>
) -> tensor<?x?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M  = tensor.dim %A, %c0 : tensor<?xf16>
  %N  = tensor.dim %B, %c1 : tensor<?x?xf16>

  // ── 计算合并后的轴长 MN = M * N ─────────────────────────
  %MN = arith.muli %M, %N : index

  // ── B: [M,N] → [M*N]，collapse两个Parallel轴 ─────────────
  %B_flat = tensor.collapse_shape %B [[0, 1]]
            : tensor<?x?xf16> into tensor<?xf16>

  // ── A: 保持[M]，访问时用 flat_idx // N 取行号 ─────────────
  // A不需要reshape，在计算body中处理广播下标

  // ── 在合并后的1D空间上计算 ─────────────────────────────
  %empty_flat = tensor.empty(%MN) : tensor<?xf16>
  %result_flat = linalg.generic {
    indexing_maps = [
      // A的访问：flat_idx -> flat_idx // N（取行号）
      // 用affine_map表达整除（符号化）
      affine_map<(d_flat)[s0] -> (d_flat floordiv s0)>,   // s0=N
      affine_map<(d_flat) -> (d_flat)>,                    // B_flat线性
      affine_map<(d_flat) -> (d_flat)>                     // Out线性
    ],
    iterator_types = ["parallel"],   // 合并后：只有1个Parallel轴
    attrs = {axis_merged = true,
             original_axes = "M,N",
             merged_symbol = "MN"}
  } ins(%A, %B_flat : tensor<?xf16>, tensor<?xf16>)
    outs(%empty_flat : tensor<?xf16>)
    // 传入符号 N 作为affine map的symbol
    // 实际实现中通过 %N 绑定到 s0
  {
  ^bb0(%a_val: f16, %b_val: f16, %out: f16):
    %sum = arith.addf %a_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?xf16>

  // ── 结果 reshape 回 [M,N] ──────────────────────────────
  %result_2d = tensor.expand_shape %result_flat [[0, 1]]
               output_shape [%M, %N]
               : tensor<?xf16> into tensor<?x?xf16>

  return %result_2d : tensor<?x?xf16>
}


// ============================================================
// STAGE 4: Tiling（TB/Tb/t 三级切分）
//          使用 Transform Dialect 驱动
//          作用于 STAGE3 产出的 1D linalg.generic
// ============================================================

// ── 4a. Transform Dialect 调度脚本 ─────────────────────────
// 描述对哪个Op做几级tiling，参数符号化

module attributes {transform.with_named_sequence} {
  transform.named_sequence @autofuse_tiling_schedule(
      %root : !transform.any_op
  ) {
    // Step1: 匹配目标Op（1D parallel generic）
    %generic = transform.structured.match
        attributes {axis_merged = true}
        in %root : (!transform.any_op) -> !transform.any_op

    // Step2: TB级切分（核间并行，每核处理TB_size个元素）
    //        TB_size 在runtime由tiling_func计算，此处用符号 %TB
    %TB_size = transform.param.constant 0 : i64  // 0表示符号化，runtime填入
    %tiled_TB, %loop_TB = transform.structured.tile_using_for %generic [%TB_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    // 标记TB循环为核间并行（映射到AiCore的BlockIdx）
    transform.loop.map_to_blocks %loop_TB {block_dims = [0]}
        : (!transform.any_op) -> ()

    // Step3: Tb级切分（UB搬运粒度）
    %Tb_size = transform.param.constant 0 : i64  // 符号化
    %tiled_Tb, %loop_Tb = transform.structured.tile_using_for %tiled_TB [%Tb_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    // Tb循环在片上串行执行（UB buffer reuse）

    // Step4: t级切分（向量化粒度，固定128个half）
    %t_size = transform.param.constant 128 : i64
    %tiled_t, %loop_t = transform.structured.tile_using_for %tiled_Tb [%t_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    // 标记t循环为向量化
    transform.structured.vectorize %tiled_t : !transform.any_op

    transform.yield
  }
}

// ── 4b. Tiling后的 Affine Dialect IR ──────────────────────
// 展示三级循环嵌套结构（具体化后的样子）

func.func @entry_after_tiling(
    %A      : memref<?xf16>,        // bufferize后用memref
    %B_flat : memref<?xf16>,
    %Out    : memref<?xf16>,
    // Tiling参数（由tiling_func在host计算后传入）
    %MN      : index,
    %TB_size : index,
    %Tb_size : index,
    %t_size  : index,
    %core_id : index               // GetBlockIdx()
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // ── TB级：本核负责的偏移 ────────────────────────────────
  %tb_offset = arith.muli %core_id, %TB_size : index

  // ── Tb级循环：分批搬运计算 ──────────────────────────────
  affine.for %tb = 0 to %TB_size step %Tb_size {
    %flat_start = arith.addi %tb_offset, %tb : index
    // 实际处理量（尾块保护）
    %remain = arith.subi %TB_size, %tb : index
    %actual_Tb = arith.minsi %Tb_size, %remain : index

    // ── 分配UB本地Buffer（片上SRAM） ──────────────────────
    %buf_A   = memref.alloca(%Tb_size) : memref<?xf16, #ascend.memory<UB>>
    %buf_B   = memref.alloca(%Tb_size) : memref<?xf16, #ascend.memory<UB>>
    %buf_out = memref.alloca(%Tb_size) : memref<?xf16, #ascend.memory<UB>>

    // ── DMA搬运 B片段：GM → UB（连续访问）────────────────
    ascend.dma_copy
        src = %B_flat[%flat_start],
        dst = %buf_B[%c0],
        size = %actual_Tb
        {src_space = "GM", dst_space = "UB"} : (index, index, index) -> ()

    // ── Broadcast搬运 A：GM → UB（按行号gather）──────────
    // 每个flat_idx对应A[flat_idx // N]
    // 用符号化affine map计算行号
    ascend.broadcast_copy
        src = %A,
        dst = %buf_A[%c0],
        flat_start = %flat_start,
        count = %actual_Tb,
        row_stride = affine_map<()[s0] -> (s0)>  // s0=N，运行时传入
        {src_space = "GM", dst_space = "UB"} : (index, index, index) -> ()

    // ── t级循环：向量化计算 ─────────────────────────────
    affine.for %t = 0 to %actual_Tb step %t_size {
      %actual_t = arith.minsi %t_size, (arith.subi %actual_Tb, %t) : index

      // 向量Add（t_size个元素并行）
      // 对应 AscendC 的 Add(dst, src0, src1, count)
      %vec_a = vector.load %buf_A[%t] : memref<?xf16,#ascend.memory<UB>>,
                                        vector<128xf16>
      %vec_b = vector.load %buf_B[%t] : memref<?xf16,#ascend.memory<UB>>,
                                        vector<128xf16>
      %vec_sum = arith.addf %vec_a, %vec_b : vector<128xf16>
      vector.store %vec_sum, %buf_out[%t] : memref<?xf16,#ascend.memory<UB>>,
                                            vector<128xf16>
    }

    // ── DMA写回：UB → GM ────────────────────────────────
    ascend.dma_copy
        src = %buf_out[%c0],
        dst = %Out[%flat_start],
        size = %actual_Tb
        {src_space = "UB", dst_space = "GM"} : (index, index, index) -> ()
  }

  return
}


// ============================================================
// STAGE 5: Tiling函数（host侧执行，符号化计算具体tiling参数）
//          输入：运行时的具体Shape M, N
//          输出：TilingData结构体各字段
// ============================================================

// TilingData 结构（用llvm.struct表示，传递给Device Kernel）
!TilingData = !llvm.struct<"TilingData", (
    i64,   // M
    i64,   // N
    i64,   // MN = M*N
    i64,   // TB_size
    i64,   // Tb_size
    i64,   // t_size
    i64    // core_num（实际使用的AiCore数量）
)>

func.func @tiling_func(
    %M : i64,
    %N : i64
) -> !TilingData {

  // ── 硬件常量 ──────────────────────────────────────────
  %UB_BYTES   = arith.constant 262144 : i64   // 256KB UB
  %CORE_NUM   = arith.constant 20 : i64       // AiCore总数
  %ELEM_BYTES = arith.constant 2 : i64        // f16 = 2字节
  %VEC_WIDTH  = arith.constant 128 : i64      // 向量计算宽度

  // ── 计算 MN ───────────────────────────────────────────
  %MN = arith.muli %M, %N : i64

  // ── t_size：固定向量宽度128 ───────────────────────────
  %t_size = arith.constant 128 : i64

  // ── Tb_size：UB能放的元素数（需要3个buffer：A/B/Out）──
  %total_elems = arith.divsi %UB_BYTES, %ELEM_BYTES : i64    // 131072个f16
  %buf_elems   = arith.divsi %total_elems, %c3 : i64         // 三等分
  // 向下对齐到 t_size
  %Tb_aligned  = arith.divsi %buf_elems, %t_size : i64
  %Tb_size     = arith.muli %Tb_aligned, %t_size : i64       // 43648*128=...

  // ── TB_size：每核分配量，按核数均分后向上对齐Tb ───────
  %c1_i64      = arith.constant 1 : i64
  %per_core    = arith.addi (arith.divsi %MN, %CORE_NUM : i64),
                             %c1_i64 : i64                   // ceil div
  %TB_div      = arith.divsi %per_core, %Tb_size : i64
  %TB_rem      = arith.remsi %per_core, %Tb_size : i64
  %TB_need_up  = arith.cmpi sgt, %TB_rem, %c0_i64 : i64
  %TB_up       = arith.select %TB_need_up, %c1_i64, %c0_i64 : i64
  %TB_size     = arith.muli (arith.addi %TB_div, %TB_up : i64),
                             %Tb_size : i64                  // align_up(per_core, Tb)

  // ── 实际使用核数 ──────────────────────────────────────
  %core_div    = arith.divsi %MN, %TB_size : i64
  %core_rem    = arith.remsi %MN, %TB_size : i64
  %core_has_r  = arith.cmpi sgt, %core_rem, %c0_i64 : i64
  %core_up     = arith.select %core_has_r, %c1_i64, %c0_i64 : i64
  %core_num    = arith.addi %core_div, %core_up : i64

  // ── 构造并返回 TilingData ─────────────────────────────
  %td_0 = llvm.mlir.undef : !TilingData
  %td_1 = llvm.insertvalue %M,       %td_0[0] : !TilingData
  %td_2 = llvm.insertvalue %N,       %td_1[1] : !TilingData
  %td_3 = llvm.insertvalue %MN,      %td_2[2] : !TilingData
  %td_4 = llvm.insertvalue %TB_size, %td_3[3] : !TilingData
  %td_5 = llvm.insertvalue %Tb_size, %td_4[4] : !TilingData
  %td_6 = llvm.insertvalue %t_size,  %td_5[5] : !TilingData
  %td_7 = llvm.insertvalue %core_num,%td_6[6] : !TilingData

  return %td_7 : !TilingData
}


// ============================================================
// STAGE 6: Runtime Dispatch（host侧选Kernel + 下发）
//          本例只有1个Kernel模板（1个合并轴）
//          若有Reduce则会有N个模板，此处演示dispatch结构
// ============================================================

func.func @runtime_dispatch(
    %A   : memref<?xf16>,
    %B   : memref<?x?xf16>,
    %Out : memref<?x?xf16>
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // ── 获取运行时Shape ────────────────────────────────────
  %M = memref.dim %A, %c0 : memref<?xf16>
  %N = memref.dim %B, %c1 : memref<?x?xf16>

  // ── 转换为i64供tiling_func使用 ────────────────────────
  %M_i64 = arith.index_cast %M : index to i64
  %N_i64 = arith.index_cast %N : index to i64

  // ── 执行host侧tiling计算 ───────────────────────────────
  %tiling = func.call @tiling_func(%M_i64, %N_i64)
            : (i64, i64) -> !TilingData

  // ── 解包TilingData ─────────────────────────────────────
  %TB_size  = llvm.extractvalue %tiling[3] : !TilingData
  %Tb_size  = llvm.extractvalue %tiling[4] : !TilingData
  %t_size   = llvm.extractvalue %tiling[5] : !TilingData
  %core_num = llvm.extractvalue %tiling[6] : !TilingData

  // ── B展平：[M,N] → [M*N] ──────────────────────────────
  %MN = arith.muli %M, %N : index
  %B_flat = memref.collapse_shape %B [[0,1]]
            : memref<?x?xf16> into memref<?xf16>
  %Out_flat = memref.collapse_shape %Out [[0,1]]
              : memref<?x?xf16> into memref<?xf16>

  // ── 核间并行下发（模拟Launch <<< core_num >>> ）────────
  // 使用 scf.parallel 表示多核并行
  %core_num_idx = arith.index_cast %core_num : i64 to index
  %TB_idx       = arith.index_cast %TB_size  : i64 to index
  %Tb_idx       = arith.index_cast %Tb_size  : i64 to index
  %t_idx        = arith.index_cast %t_size   : i64 to index
  %MN_idx       = arith.index_cast
                    (llvm.extractvalue %tiling[2] : !TilingData)
                    : i64 to index

  scf.parallel (%core_id) = (%c0) to (%core_num_idx) step (%c1) {
    func.call @entry_after_tiling(
        %A, %B_flat, %Out_flat,
        %MN_idx,
        %TB_idx, %Tb_idx, %t_idx,
        %core_id
    ) : (memref<?xf16>, memref<?xf16>, memref<?xf16>,
         index, index, index, index, index) -> ()
    scf.reduce
  }

  return
}


// ============================================================
// STAGE 7: 最终 AscendC Lowering 后的样子（目标代码伪表示）
//
// 从 Affine/Vector Dialect lower到 Ascend Dialect后，
// 最终生成类似 AscendC 的 Kernel 代码：
//
//   __global__ __aicore__ void fused_kernel(
//       GM_ADDR A, GM_ADDR B, GM_ADDR Out, TilingData* tiling
//   ) {
//       int core_id = GetBlockIdx();
//       int64_t tb_offset = core_id * tiling->TB_size;
//
//       for (int64_t tb = 0; tb < tiling->TB_size; tb += tiling->Tb_size) {
//           int64_t flat_start = tb_offset + tb;
//           int64_t actual_Tb  = min(tiling->Tb_size,
//                                    tiling->TB_size - tb);
//
//           // GM→UB：B连续搬运
//           DataCopy(buf_B, B[flat_start], actual_Tb);
//
//           // GM→UB：A广播搬运（按行号）
//           BroadcastCopy(buf_A, A, flat_start, actual_Tb, tiling->N);
//
//           // 向量计算
//           for (int64_t t = 0; t < actual_Tb; t += 128) {
//               Add(buf_out[t], buf_A[t], buf_B[t], 128);
//           }
//
//           // UB→GM：写回
//           DataCopy(Out[flat_start], buf_out, actual_Tb);
//       }
//   }
// ============================================================


// ============================================================
// 附：如果加入 ReduceSum(axis=1)，轴分组变化示意
//
//   Add输出 D[M,N] → ReduceSum(d1) → E[M]
//
//   全局轴分析：
//     d0 = Parallel（所有Op均Parallel）
//     d1 = Reduce  （ReduceSum消除d1）
//
//   → 轴分组不一致（有Reduce），需要分段处理
//   → 生成 2 个 Kernel 模板：
//
//   模板A（Reduce轴在内层）：
//     for d0 (Parallel, 核间切分):
//       for d1 (Reduce, 片内规约):
//         ...
//     TilingData_A = {TB_M, Tb_M, t_M, full_N}
//
//   模板B（Reduce轴外提，适合N很大时）：
//     for d1_outer (Reduce分块):
//       for d0 (Parallel):
//         for d1_inner:
//           ...
//     TilingData_B = {TB_M, Tb_N, t_N, split_N}
//
//   Runtime：根据 M/N 比例用性能模型选择模板A或B
// ============================================================
