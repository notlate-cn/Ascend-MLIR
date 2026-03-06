// ================================================================
//  AscendNPU: Matmul + Add + Log 完整MLIR编译流程
//
//  计算语义：
//    A: [M, K]   B: [K, N]   Bias: [N]
//    C = Matmul(A, B)          → [M, N]   (Cube单元)
//    D = Add(C, Bias)          → [M, N]   (Vector单元)
//    E = Log(D)                → [M, N]   (Vector单元)
//
//  核心挑战：
//    Matmul  → AscendNPU Cube单元  (L0A/L0B/L0C内存)
//    Add/Log → AscendNPU Vector单元 (UB内存)
//    两个单元的内存空间不同，需要显式搬运 L0C→UB
//    但融合后可以消除 C 的 GM 落盘，显著减少带宽
//
//  编译阶段：
//    Stage 0: 原始高层IR
//    Stage 1: 轴分组 & 融合判定
//    Stage 2: 融合子图IR‰
//    Stage 3: 轴分析 & Tiling决策
//    Stage 4: 自定义Ascend Dialect下沉
//    Stage 5: Tiling函数（host侧）
//    Stage 6: 双Buffer流水 & Kernel模板
//    Stage 7: Runtime Dispatch
// ================================================================


// ================================================================
// STAGE 0: 原始 High-Level IR
//          Linalg命名Op + Tensor值语义 + 完全符号化Shape
// ================================================================

func.func @entry_before_fusion(
    %A    : tensor<?x?xf16>,    // [M, K]
    %B    : tensor<?x?xf16>,    // [K, N]
    %Bias : tensor<?xf16>       // [N]
) -> tensor<?x?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf16>   // 符号 M
  %K = tensor.dim %A, %c1 : tensor<?x?xf16>   // 符号 K
  %N = tensor.dim %B, %c1 : tensor<?x?xf16>   // 符号 N

  // ── Op1: Matmul ─────────────────────────────────────────
  // linalg.matmul 是具名Op，自带语义
  // iterator_types: [parallel, parallel, reduction]
  //   d0=M → Parallel
  //   d1=N → Parallel
  //   d2=K → Reduction（收缩轴）
  %empty_C = tensor.empty(%M, %N) : tensor<?x?xf16>
  %zero    = arith.constant 0.0 : f16
  %C_init  = linalg.fill ins(%zero : f16)
                         outs(%empty_C : tensor<?x?xf16>) -> tensor<?x?xf16>
  %C = linalg.matmul
       ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
       outs(%C_init : tensor<?x?xf16>) -> tensor<?x?xf16>

  // ── Op2: Add (Bias广播) ──────────────────────────────────
  // Bias[N] 沿 M 轴广播后与 C[M,N] 相加
  // indexing_maps体现广播：Bias只用d1(N)，不用d0(M)
  %empty_D = tensor.empty(%M, %N) : tensor<?x?xf16>
  %D = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // C: 正常二维读
      affine_map<(d0, d1) -> (d1)>,       // Bias: 只用N轴，M轴广播
      affine_map<(d0, d1) -> (d0, d1)>   // D: 输出
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C, %Bias : tensor<?x?xf16>, tensor<?xf16>)
    outs(%empty_D : tensor<?x?xf16>) {
  ^bb0(%c_val: f16, %bias_val: f16, %out: f16):
    %sum = arith.addf %c_val, %bias_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  // ── Op3: Log ────────────────────────────────────────────
  // 逐元素 Log，纯Elementwise
  %empty_E = tensor.empty(%M, %N) : tensor<?x?xf16>
  %E = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%D : tensor<?x?xf16>)
    outs(%empty_E : tensor<?x?xf16>) {
  ^bb0(%d_val: f16, %out: f16):
    %log_val = math.log %d_val : f16
    linalg.yield %log_val : f16
  } -> tensor<?x?xf16>

  return %E : tensor<?x?xf16>
}


// ================================================================
// STAGE 1: 轴分组分析 & 融合判定
//
// 全局轴（补齐到rank=3，Matmul有3个逻辑轴）：
//   d0 = M (行)
//   d1 = N (列)
//   d2 = K (收缩轴)
//
// 各Op轴分组：
// ┌──────────┬──────┬──────┬───────┬──────────────────────────┐
// │ Op       │  d0  │  d1  │  d2   │ 说明                     │
// ├──────────┼──────┼──────┼───────┼──────────────────────────┤
// │ Matmul   │  P   │  P   │   R   │ K轴规约                  │
// │ Add(Bias)│  P   │  P   │  (1)  │ d2不存在，视为已消除     │
// │ Log      │  P   │  P   │  (1)  │ 同上                     │
// └──────────┴──────┴──────┴───────┴──────────────────────────┘
//
// 融合判定：
//   d0: 全Parallel → OK
//   d1: 全Parallel → OK
//   d2: 只有Matmul有Reduce，Add/Log在该轴上size=1（不参与）
//   → 关键规则：后续Op不在Reduce轴上做任何计算
//             → Matmul的Reduce是"内部"的，输出已消除d2
//             → Add/Log仅作用于Matmul输出的[M,N]维度
//   → 融合判定：可融合 ✅
//     策略：Matmul单独用Cube，Add+Log用Vector，L0C→UB衔接
//
// 融合模式：Cube-Vector 混合融合（NPU特有）
//   └─ 不能合并为单个linalg.generic（计算单元不同）
//   └─ 而是保持两段，通过片上Buffer直接传递，消除GM落盘
// ================================================================


// ================================================================
// STAGE 2: 融合后子图 IR
//          Matmul保持独立（Cube语义），Add+Log合并为Vector段
//          中间结果C不再落GM，通过自定义attr标记片上传递
// ================================================================

func.func @entry_after_fusion(
    %A    : tensor<?x?xf16>,
    %B    : tensor<?x?xf16>,
    %Bias : tensor<?xf16>
) -> tensor<?x?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %A, %c0 : tensor<?x?xf16>
  %N = tensor.dim %B, %c1 : tensor<?x?xf16>

  // ── Cube段：Matmul ──────────────────────────────────────
  // 标记 compute_unit = "cube"，结果留在 L0C（不落GM）
  %empty_C = tensor.empty(%M, %N) : tensor<?x?xf16>
  %zero    = arith.constant 0.0 : f16
  %C_init  = linalg.fill ins(%zero : f16)
                         outs(%empty_C : tensor<?x?xf16>) -> tensor<?x?xf16>
  %C = linalg.matmul {
    lowering_config = #ascend.lowering_config<
      compute_unit = "cube",
      output_buffer = "L0C",       // 输出留在L0C，不写GM
      tile_m = 128, tile_n = 128, tile_k = 64   // 符号化后由tiling_func填
    >
  } ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
    outs(%C_init : tensor<?x?xf16>) -> tensor<?x?xf16>

  // ── Vector段：Add + Log 融合为单个 generic ──────────────
  // 标记 compute_unit = "vector"，输入来自L0C（隐式DMA L0C→UB）
  %empty_E = tensor.empty(%M, %N) : tensor<?x?xf16>
  %E = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // C (来自L0C)
      affine_map<(d0, d1) -> (d1)>,       // Bias (来自GM，广播)
      affine_map<(d0, d1) -> (d0, d1)>   // E (输出到GM)
    ],
    iterator_types = ["parallel", "parallel"],
    lowering_config = #ascend.lowering_config<
      compute_unit = "vector",
      input_buffer = "L0C→UB",    // 标记C从L0C搬入UB
      fused_ops = ["add_bias", "log"]
    >
  } ins(%C, %Bias : tensor<?x?xf16>, tensor<?xf16>)
    outs(%empty_E : tensor<?x?xf16>) {
  ^bb0(%c_val: f16, %bias_val: f16, %out: f16):
    %add_val = arith.addf %c_val, %bias_val : f16
    %log_val = math.log %add_val : f16
    linalg.yield %log_val : f16
  } -> tensor<?x?xf16>

  return %E : tensor<?x?xf16>
}


// ================================================================
// STAGE 3: 轴分析 & 分级 Tiling 决策
//
// Matmul轴：[d0=M, d1=N, d2=K]
//   → 三级切分对应AscendNPU Cube流水：
//
//   TB级（核间）：
//     切 d0_TB × d1_TB 块分配给各AiCore
//     典型值：d0_TB=128行, d1_TB=128列
//
//   Tb级（L1 → L0 流水）：
//     切 d0_Tb × d1_Tb × d2_Tb
//     d0_Tb=128, d1_Tb=128, d2_Tb=64 → 一次MTE2搬运量
//
//   t级（L0C → UB，Cube最小计算粒度）：
//     16×16 的基础矩阵块（fp16：16×16×16）
//
// Vector段轴：[d0=M, d1=N]（与Matmul输出对齐）
//   → 跟随Matmul的TB/Tb切分
//   → t级：128个half（向量化宽度）
//
// 关键约束：
//   Vector段的 Tb_size = Matmul的 d0_Tb × d1_Tb
//   保证 L0C 的一块结果正好填满 UB 的一个处理单元
// ================================================================


// ================================================================
// STAGE 4: 自定义 Ascend Dialect IR
//          显式建模NPU内存层次和Cube/Vector指令
// ================================================================

// ── 4a. 内存空间定义（Ascend Dialect 属性）─────────────────
// #ascend.memory<GM>   全局内存 (HBM)
// #ascend.memory<L1>   一级缓存 (AI Core片上, ~1MB)
// #ascend.memory<L0A>  矩阵A缓冲 (专用, ~64KB)
// #ascend.memory<L0B>  矩阵B缓冲 (专用, ~64KB)
// #ascend.memory<L0C>  矩阵C累加 (专用, ~256KB)
// #ascend.memory<UB>   统一Buffer (Vector使用, ~256KB)

// ── 4b. Tiling后的核心计算函数（单个AiCore视角）──────────────

func.func @matmul_add_log_one_core(
    // GM 指针
    %A_gm    : memref<?x?xf16, #ascend.memory<GM>>,   // [M, K]
    %B_gm    : memref<?x?xf16, #ascend.memory<GM>>,   // [K, N]
    %Bias_gm : memref<?xf16,   #ascend.memory<GM>>,   // [N]
    %E_gm    : memref<?x?xf16, #ascend.memory<GM>>,   // [M, N] 输出

    // Tiling参数（由tiling_func在host计算）
    %core_id  : index,
    %M_tb : index, %N_tb : index,       // 本核负责的M/N范围大小
    %m_off: index, %n_off: index,       // 本核在全局M/N的偏移
    %K_total  : index,                  // K总长
    %d0_Tb: index, %d1_Tb: index,      // L0级Matmul块大小
    %d2_Tb: index,                      // K方向分块
    %vec_Tb   : index                   // Vector段UB处理粒度
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // ── 分配片上Buffer（静态大小，tiling保证不超UB） ──────────
  // Cube专用Buffer
  %L0A = memref.alloc() : memref<128x64xf16,  #ascend.memory<L0A>>  // A块
  %L0B = memref.alloc() : memref<64x128xf16,  #ascend.memory<L0B>>  // B块
  %L0C = memref.alloc() : memref<128x128xf16, #ascend.memory<L0C>>  // 累加C

  // Vector专用Buffer（双Buffer，用于流水线）
  %UB_C0  = memref.alloc() : memref<128x128xf16, #ascend.memory<UB>> // pingpong 0
  %UB_C1  = memref.alloc() : memref<128x128xf16, #ascend.memory<UB>> // pingpong 1
  %UB_out = memref.alloc() : memref<128x128xf16, #ascend.memory<UB>> // 结果
  %UB_bias= memref.alloc() : memref<128xf16,     #ascend.memory<UB>> // Bias

  // ── 预加载 Bias（只需加载一次，沿M轴复用） ────────────────
  ascend.mte2 {  // MTE2 = GM→UB 搬运引擎
    src = %Bias_gm[%n_off],
    dst = %UB_bias[%c0],
    size = %N_tb
  } : (index, index, index) -> ()

  // ================================================================
  //  主循环：沿 K 轴分块做 Matmul，L0C累加
  //  每块 [d0_Tb=128, d2_Tb=64] × [d2_Tb=64, d1_Tb=128]
  // ================================================================
  %K_steps = arith.ceildivsi %K_total, %d2_Tb : index

  scf.for %k_step = %c0 to %K_steps step %c1 {
    %k_off = arith.muli %k_step, %d2_Tb : index

    // ── MTE1：GM → L1 → L0A（A块搬运）────────────────────
    // MTE1 = L1→L0A 精细搬运（通常先GM→L1再L1→L0A）
    ascend.mte1 {
      src_gm  = %A_gm[%m_off, %k_off],
      dst_l0a = %L0A[%c0, %c0],
      rows = %d0_Tb,   // 128
      cols = %d2_Tb    // 64
    } : (index, index, index, index) -> ()

    // ── MTE1：GM → L1 → L0B（B块搬运）────────────────────
    ascend.mte1 {
      src_gm  = %B_gm[%k_off, %n_off],
      dst_l0b = %L0B[%c0, %c0],
      rows = %d2_Tb,   // 64
      cols = %d1_Tb    // 128
    } : (index, index, index, index) -> ()

    // ── Cube计算：L0A × L0B → L0C（累加）─────────────────
    // 对应 AscendC 的 Mmad 指令
    // 16×16×16 为最小计算粒度（fp16）
    %is_first = arith.cmpi eq, %k_step, %c0 : index
    ascend.cube.mmad {
      A    = %L0A,
      B    = %L0B,
      C    = %L0C,
      M    = %d0_Tb,    // 128
      N    = %d1_Tb,    // 128
      K    = %d2_Tb,    // 64
      init = %is_first  // 第一次清零，后续累加
    } : (memref<128x64xf16,  #ascend.memory<L0A>>,
         memref<64x128xf16,  #ascend.memory<L0B>>,
         memref<128x128xf16, #ascend.memory<L0C>>,
         index, index, index, i1) -> ()
    // 同步：等待Cube计算完成
    ascend.sync {scope = "cube"} : () -> ()
  }
  // K轴循环结束，L0C中存有完整的 C[m_off:m_off+128, n_off:n_off+128]

  // ================================================================
  //  L0C → UB 搬运 + Vector计算（Add + Log）
  //  沿 M 方向分小块流水，Vector_Tb 行为一组
  // ================================================================
  // 使用双Buffer（Ping-Pong）：搬运下一块的同时计算当前块
  %vec_steps = arith.ceildivsi %M_tb, %vec_Tb : index
  %ping = arith.constant 0 : i32   // 当前计算用哪个UB Buffer

  scf.for %v_step = %c0 to %vec_steps step %c1 {
    %v_off       = arith.muli %v_step, %vec_Tb : index
    %ping_buf    = scf.if %is_ping -> (memref<128x128xf16, #ascend.memory<UB>>) {
                     scf.yield %UB_C0 : memref<128x128xf16, #ascend.memory<UB>>
                   } else {
                     scf.yield %UB_C1 : memref<128x128xf16, #ascend.memory<UB>>
                   }

    // ── MTE3：L0C → UB（当前块搬运）──────────────────────
    // MTE3 = L0C→UB 专用搬运引擎（Cube→Vector 桥梁）
    ascend.mte3 {
      src_l0c = %L0C[%v_off, %c0],
      dst_ub  = %ping_buf[%c0, %c0],
      rows    = %vec_Tb,      // 当前搬多少行（尾块保护）
      cols    = %N_tb         // N列全搬
    } : (index, index, index, index) -> ()

    // ── Vector计算：Add Bias ───────────────────────────────
    // 对 ping_buf 的每行加 UB_bias（广播）
    // 对应 AscendC: Add(dst, src, bias, count)
    ascend.vector.binary<"add"> {
      src0   = %ping_buf,
      src1   = %UB_bias,      // N维Bias，自动沿行广播
      dst    = %UB_out,
      rows   = %vec_Tb,
      cols   = %N_tb,
      broadcast_src1 = true   // src1在行方向广播
    } : (memref<128x128xf16, #ascend.memory<UB>>,
         memref<128xf16,     #ascend.memory<UB>>,
         memref<128x128xf16, #ascend.memory<UB>>,
         index, index) -> ()

    // ── Vector计算：Log ────────────────────────────────────
    // 逐元素 Log，对应 AscendC: Ln(dst, src, count)
    ascend.vector.unary<"log"> {
      src  = %UB_out,
      dst  = %UB_out,         // in-place
      rows = %vec_Tb,
      cols = %N_tb
    } : (memref<128x128xf16, #ascend.memory<UB>>,
         memref<128x128xf16, #ascend.memory<UB>>,
         index, index) -> ()

    // ── MTE2：UB → GM（写回结果）─────────────────────────
    %e_row = arith.addi %m_off, %v_off : index
    ascend.mte2 {
      src_ub = %UB_out[%c0, %c0],
      dst_gm = %E_gm[%e_row, %n_off],
      rows   = %vec_Tb,
      cols   = %N_tb
    } : (index, index, index, index) -> ()

    // Ping-Pong切换（下次迭代用另一个Buffer）
    // scf.for 的下一轮 MTE3 可与本轮 Vector 计算重叠
  }

  // 释放片上Buffer
  memref.dealloc %L0A  : memref<128x64xf16,  #ascend.memory<L0A>>
  memref.dealloc %L0B  : memref<64x128xf16,  #ascend.memory<L0B>>
  memref.dealloc %L0C  : memref<128x128xf16, #ascend.memory<L0C>>
  memref.dealloc %UB_C0  : memref<128x128xf16, #ascend.memory<UB>>
  memref.dealloc %UB_C1  : memref<128x128xf16, #ascend.memory<UB>>
  memref.dealloc %UB_out : memref<128x128xf16, #ascend.memory<UB>>
  memref.dealloc %UB_bias: memref<128xf16,     #ascend.memory<UB>>

  return
}


// ================================================================
// STAGE 5: Tiling函数（host侧执行，符号化 → 具体值）
//
// 目标：根据 M/N/K 计算各级分块大小，满足：
//   1. L0A + L0B + L0C 不超片上容量
//   2. UB 不超容量（需放 C块 + Bias + Out）
//   3. 块大小对齐到 Cube 最小粒度（16×16）
// ================================================================

!MatmulTilingData = !llvm.struct<"MatmulTilingData", (
    i64,   // M
    i64,   // N
    i64,   // K
    i64,   // core_num（实际使用AiCore数）
    i64,   // M_tb（每核负责的M行数）
    i64,   // N_tb（每核负责的N列数，通常=全N）
    i64,   // d0_Tb = 128（Matmul M方向块，固定对齐Cube粒度）
    i64,   // d1_Tb = 128（Matmul N方向块）
    i64,   // d2_Tb = 64 （Matmul K方向块）
    i64    // vec_Tb（Vector段处理行数，受UB大小约束）
)>

func.func @tiling_func_matmul(
    %M : i64, %N : i64, %K : i64
) -> !MatmulTilingData {

  // ── 硬件常量 ──────────────────────────────────────────
  %CORE_NUM    = arith.constant 20   : i64
  %L0A_BYTES   = arith.constant 65536  : i64   // 64KB
  %L0B_BYTES   = arith.constant 65536  : i64   // 64KB
  %L0C_BYTES   = arith.constant 262144 : i64   // 256KB
  %UB_BYTES    = arith.constant 262144 : i64   // 256KB
  %ELEM_BYTES  = arith.constant 2      : i64   // f16=2字节
  %CUBE_BASE   = arith.constant 16     : i64   // Cube最小粒度16
  %c1          = arith.constant 1      : i64

  // ── Cube分块（受L0A/L0B/L0C容量约束）─────────────────
  // L0A 存 [d0_Tb, d2_Tb]，L0B 存 [d2_Tb, d1_Tb]
  // 约束：d0_Tb * d2_Tb * 2 ≤ L0A_BYTES  → 128*64*2 = 16384 ≤ 65536 ✅
  //       d2_Tb * d1_Tb * 2 ≤ L0B_BYTES  → 64*128*2 = 16384 ≤ 65536 ✅
  //       d0_Tb * d1_Tb * 4 ≤ L0C_BYTES  → 128*128*4=65536 ≤ 262144 ✅ (fp32累加)
  %d0_Tb = arith.constant 128 : i64
  %d1_Tb = arith.constant 128 : i64
  %d2_Tb = arith.constant 64  : i64

  // ── 核间分块：沿 M 轴切分给各核 ────────────────────────
  // 每核负责 M_tb 行，N轴不切（每核做完整N列）
  // M_tb 向上对齐到 d0_Tb=128
  %M_per_core_raw = arith.ceildivsi %M, %CORE_NUM : i64
  %M_align_mask   = arith.subi %d0_Tb, %c1 : i64
  %M_tb           = arith.andi
                      (arith.addi %M_per_core_raw, %M_align_mask : i64),
                      (arith.xori %M_align_mask,
                        arith.constant -1 : i64 : i64) : i64
  // 简化写法：M_tb = align_up(M/CORE_NUM, 128)

  // 实际核数（尾部可能有空核）
  %core_num = arith.ceildivsi %M, %M_tb : i64

  // ── Vector段 UB 分块 ──────────────────────────────────
  // UB需要存放：
  //   UB_C (从L0C搬来的C块)：vec_Tb * N * 2字节
  //   UB_bias：N * 2字节
  //   UB_out：vec_Tb * N * 2字节（复用UB_C的pingpong）
  // 双Buffer：UB_C有2份，实际: 2*vec_Tb*N*2 + N*2 ≤ UB_BYTES
  // → vec_Tb ≤ (UB_BYTES/2 - N*2) / (N*2)
  %UB_half       = arith.divsi %UB_BYTES, (arith.constant 2 : i64) : i64
  %bias_bytes    = arith.muli %N, %ELEM_BYTES : i64
  %avail_for_C   = arith.subi %UB_half, %bias_bytes : i64
  %row_bytes      = arith.muli %N, %ELEM_BYTES : i64
  %vec_Tb_raw    = arith.divsi %avail_for_C, %row_bytes : i64
  // 向下对齐到 CUBE_BASE=16（保证L0C搬运对齐）
  %vec_Tb        = arith.muli
                     (arith.divsi %vec_Tb_raw, %CUBE_BASE : i64),
                     %CUBE_BASE : i64

  // ── 构造 TilingData ────────────────────────────────────
  %td = llvm.mlir.undef : !MatmulTilingData
  %td1 = llvm.insertvalue %M,        %td[0]  : !MatmulTilingData
  %td2 = llvm.insertvalue %N,        %td1[1] : !MatmulTilingData
  %td3 = llvm.insertvalue %K,        %td2[2] : !MatmulTilingData
  %td4 = llvm.insertvalue %core_num, %td3[3] : !MatmulTilingData
  %td5 = llvm.insertvalue %M_tb,     %td4[4] : !MatmulTilingData
  %td6 = llvm.insertvalue %N,        %td5[5] : !MatmulTilingData  // N_tb=全N
  %td7 = llvm.insertvalue %d0_Tb,    %td6[6] : !MatmulTilingData
  %td8 = llvm.insertvalue %d1_Tb,    %td7[7] : !MatmulTilingData
  %td9 = llvm.insertvalue %d2_Tb,    %td8[8] : !MatmulTilingData
  %tdA = llvm.insertvalue %vec_Tb,   %td9[9] : !MatmulTilingData

  return %tdA : !MatmulTilingData
}


// ================================================================
// STAGE 6: 双Buffer流水线展开（Pipeline）
//          Cube 计算 与 MTE 搬运 重叠执行
//
//  时间线（理想状态）：
//
//  MTE1(k=0)  ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
//  Cube(k=0)  ░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░
//  MTE1(k=1)  ░░░░░░░▓▓▓░░░░░░░░░░░░░░░░░░░░░░░  ← 与Cube(k=0)重叠
//  Cube(k=1)  ░░░░░░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░
//  ...（K步循环）
//  MTE3(v=0)  ░░░░░░░░░░░░░░░░░▓▓░░░░░░░░░░░░░░  ← K循环结束后
//  Vector(v=0)░░░░░░░░░░░░░░░░░░░▓▓▓░░░░░░░░░░░
//  MTE3(v=1)  ░░░░░░░░░░░░░░░░░░░░░▓▓░░░░░░░░░  ← 与Vector(v=0)重叠
//  Vector(v=1)░░░░░░░░░░░░░░░░░░░░░░░▓▓▓░░░░░░
//  MTE2(v=0)  ░░░░░░░░░░░░░░░░░░░░░░░░▓░░░░░░░  ← 写回
//
//  通过 ascend.pipeline 属性标记，lowering时自动展开流水

func.func @matmul_add_log_pipelined(
    %A_gm    : memref<?x?xf16, #ascend.memory<GM>>,
    %B_gm    : memref<?x?xf16, #ascend.memory<GM>>,
    %Bias_gm : memref<?xf16,   #ascend.memory<GM>>,
    %E_gm    : memref<?x?xf16, #ascend.memory<GM>>,
    %tiling  : !MatmulTilingData
) {
  // 解包tiling参数
  %core_id = ascend.get_block_idx : index
  %M_tb    = llvm.extractvalue %tiling[4] : !MatmulTilingData
  %d0_Tb   = llvm.extractvalue %tiling[6] : !MatmulTilingData
  %d1_Tb   = llvm.extractvalue %tiling[7] : !MatmulTilingData
  %d2_Tb   = llvm.extractvalue %tiling[8] : !MatmulTilingData
  %vec_Tb  = llvm.extractvalue %tiling[9] : !MatmulTilingData
  %K       = llvm.extractvalue %tiling[2] : !MatmulTilingData
  %N       = llvm.extractvalue %tiling[1] : !MatmulTilingData

  %m_off = arith.muli %core_id,
             (arith.index_cast %M_tb : i64 to index) : index

  // ── 带流水标记的 K 轴主循环 ─────────────────────────────
  scf.for %k_step = 0 to %K step %d2_Tb
      iter_args(/*ping-pong state*/)
      {pipeline.depth = 2} {      // 双Buffer，深度2

    // Stage A：搬运（MTE1，异步）
    ascend.async.mte1 { ... }     // 不阻塞，立即返回

    // Stage B：等待上一步搬运完成，启动Cube
    ascend.sync {scope = "mte1"}
    ascend.cube.mmad { ... }

    // lowering 时自动将 Stage A(k+1) 与 Stage B(k) 重叠
    scf.yield
  }

  // ── 带流水标记的 Vector 段循环 ──────────────────────────
  scf.for %v_step = 0 to %M_tb step %vec_Tb
      {pipeline.depth = 2} {

    // Stage P：MTE3 L0C→UB（异步）
    ascend.async.mte3 { ... }

    // Stage Q：等待MTE3完成，执行 Add+Log
    ascend.sync {scope = "mte3"}
    ascend.vector.binary<"add"> { ... }
    ascend.vector.unary<"log">  { ... }

    // Stage R：MTE2 UB→GM（与下一次MTE3重叠）
    ascend.async.mte2 { ... }

    scf.yield
  }
  ascend.sync {scope = "all"}  // 等所有异步操作完成
  return
}


// ================================================================
// STAGE 7: Runtime Dispatch（host侧完整流程）
// ================================================================

func.func @runtime_dispatch_matmul(
    %A_gm    : memref<?x?xf16, #ascend.memory<GM>>,
    %B_gm    : memref<?x?xf16, #ascend.memory<GM>>,
    %Bias_gm : memref<?xf16,   #ascend.memory<GM>>,
    %E_gm    : memref<?x?xf16, #ascend.memory<GM>>
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // ── 读取运行时Shape ────────────────────────────────────
  %M = memref.dim %A_gm, %c0 : memref<?x?xf16, #ascend.memory<GM>>
  %K = memref.dim %A_gm, %c1 : memref<?x?xf16, #ascend.memory<GM>>
  %N = memref.dim %B_gm, %c1 : memref<?x?xf16, #ascend.memory<GM>>
  %M_i64 = arith.index_cast %M : index to i64
  %K_i64 = arith.index_cast %K : index to i64
  %N_i64 = arith.index_cast %N : index to i64

  // ── Host执行tiling函数（纯CPU计算，微秒级）───────────────
  %tiling = func.call @tiling_func_matmul(%M_i64, %N_i64, %K_i64)
            : (i64, i64, i64) -> !MatmulTilingData

  // ── 解包核数 ──────────────────────────────────────────
  %core_num_i64 = llvm.extractvalue %tiling[3] : !MatmulTilingData
  %core_num     = arith.index_cast %core_num_i64 : i64 to index

  // ── 核间并行下发（每核处理M_tb行） ────────────────────
  // 示例：M=1024, N=512, K=256
  //   M_tb = align_up(1024/20, 128) = align_up(52, 128) = 128
  //   core_num = ceil(1024/128) = 8
  //   d2_Tb = 64, K_steps = 256/64 = 4
  //   vec_Tb = (131072 - 512*2) / (512*2) = 127 → align_down 112
  //
  // 各核分工：
  //   Core 0: A[0:128,   :], B[:,:]  → E[0:128,   :]
  //   Core 1: A[128:256, :], B[:,:]  → E[128:256, :]
  //   ...
  //   Core 7: A[896:1024,:], B[:,:]  → E[896:1024,:]
  scf.parallel (%core_id) = (%c0) to (%core_num) step (%c1) {
    func.call @matmul_add_log_pipelined(
        %A_gm, %B_gm, %Bias_gm, %E_gm, %tiling
    ) : (memref<?x?xf16, #ascend.memory<GM>>,
         memref<?x?xf16, #ascend.memory<GM>>,
         memref<?xf16,   #ascend.memory<GM>>,
         memref<?x?xf16, #ascend.memory<GM>>,
         !MatmulTilingData) -> ()
    scf.reduce
  }

  return
}


// ================================================================
// 附录：与纯Vector融合（如上例Bcast+Add）的关键差异对比
//
// ┌──────────────────┬─────────────────┬──────────────────────┐
// │ 维度             │ Bcast+Add案例   │ Matmul+Add+Log案例   │
// ├──────────────────┼─────────────────┼──────────────────────┤
// │ 计算单元         │ 纯Vector        │ Cube + Vector 混合   │
// │ 融合方式         │ 单个linalg.gen  │ 两段（Cube段+Vec段） │
// │ 核心内存搬运     │ GM→UB→GM        │ GM→L0→L0C→UB→GM     │
// │ 融合收益         │ 减少中间GM写    │ 消除C的GM落盘        │
// │ Reduce轴         │ 无              │ K轴在Cube段规约      │
// │ Tiling约束       │ UB容量          │ L0A+L0B+L0C+UB联合  │
// │ 流水线           │ 简单单Buffer    │ Cube/Vector双Buffer  │
// │ 轴分组结论       │ 全Parallel      │ Parallel+Reduce混合  │
// │ MLIR融合判定     │ 直接fuse        │ Cube内Reduce，       │
// │                  │                 │ 后接Parallel → 可融合│
// └──────────────────┴─────────────────┴──────────────────────┘
//
// 融合合法性的统一规则：
//   后续Op（Add/Log）只作用于Matmul输出的[M,N]维度
//   Matmul的Reduce轴K对后续Op"透明"（已被消除）
//   → 后续Op视角：只有Parallel轴 → 可融合
//   → 实现方式：Matmul Cube段完成后，L0C数据直传Vector段
//              不经过GM，消除带宽瓶颈
// ================================================================
