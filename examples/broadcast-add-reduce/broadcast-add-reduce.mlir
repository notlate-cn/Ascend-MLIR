// ============================================================
// AutoFuse 完整MLIR表示 - 广播加法归约示例
// 计算图：A[M] → Broadcast → C[M,N]
//         C[M,N] + B[M,N] → D[M,N] (逐元素加法)
//         D[M,N] → ReduceSum(axis=1) → E[M] (按行求和)
//
// 本示例展示了一个包含归约操作的完整编译流程：
//   Broadcast + Add + ReduceSum → 算子融合 → Tiling → AscendC代码生成
//
// 与 broadcast-add.mlir 的区别：
//   - 本例包含 ReduceSum 归约操作
//   - d0 轴是 Parallel，d1 轴是 Reduction
//   - 需要特殊的归约轴处理策略
//   - 无法进行轴合并（因为有 Reduction 轴）
//
// 编译流程：
//   STAGE 0: High-Level IR (tensor/linalg，完全符号化)
//   STAGE 1: 轴分组分析 (d0=Parallel, d1=Reduction)
//   STAGE 2: 融合判定 (可融合为单个 Kernel)
//   STAGE 3: Tiling (TB/Tb，沿 Parallel 轴分块)
//   STAGE 4: TilingData 结构 + tiling_func
//   STAGE 5: Runtime Dispatch (多核下发)
//   STAGE 6: AscendC Lowering
// ============================================================


// ============================================================
// STAGE 0: 输入的 High-Level IR（完全符号化，M/N 动态Shape）
// ============================================================
//
//   A : tensor<?xf16>      shape=[M]      Load1
//   B : tensor<?x?xf16>    shape=[M,N]    Load2
//   E : tensor<?xf16>      shape=[M]      输出
//
//   Op1: Broadcast  A[M] → C[M,N]
//   Op2: Add        D[M,N] = C[M,N] + B[M,N]
//   Op3: ReduceSum  E[M] = sum(D, axis=1)

func.func @stage0_input(
    %input_a : tensor<?xf16>,       // shape [M]
    %input_b : tensor<?x?xf16>      // shape [M, N]
) -> tensor<?xf16> {

  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %dim_m  = tensor.dim %input_a, %idx_0 : tensor<?xf16>
  %dim_n  = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

  // ── Op1: Broadcast ──────────────────────────────────────
  // A[M] → C[M,N]，沿 d1(N轴) 广播
  %empty_c = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
  %tensor_c = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A：只访问d0，d1被广播
      affine_map<(d0, d1) -> (d0, d1)>   // C：输出
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input_a : tensor<?xf16>)
    outs(%empty_c : tensor<?x?xf16>) {
  ^bb0(%a_val: f16, %c_out: f16):
    linalg.yield %a_val : f16
  } -> tensor<?x?xf16>

  // ── Op2: Add ────────────────────────────────────────────
  // D[M,N] = C[M,N] + B[M,N]
  %empty_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
  %tensor_d = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // C
      affine_map<(d0, d1) -> (d0, d1)>,  // B
      affine_map<(d0, d1) -> (d0, d1)>   // D
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%tensor_c, %input_b : tensor<?x?xf16>, tensor<?x?xf16>)
    outs(%empty_d : tensor<?x?xf16>) {
  ^bb0(%c_val: f16, %b_val: f16, %d_out: f16):
    %sum = arith.addf %c_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  // ── Op3: ReduceSum(axis=1) ──────────────────────────────
  // E[M] = sum(D, axis=1)
  // d0 是 Parallel（输出行），d1 是 Reduction（被求和消除）
  %zero = arith.constant 0.0 : f16
  %empty_e = tensor.empty(%dim_m) : tensor<?xf16>
  %init_e = linalg.fill ins(%zero : f16) outs(%empty_e : tensor<?xf16>) -> tensor<?xf16>
  %tensor_e = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // D：完整访问
      affine_map<(d0, d1) -> (d0)>       // E：只输出d0，d1被归约
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%tensor_d : tensor<?x?xf16>)
    outs(%init_e : tensor<?xf16>) {
  ^bb0(%d_val: f16, %acc: f16):
    %new_acc = arith.addf %acc, %d_val : f16
    linalg.yield %new_acc : f16
  } -> tensor<?xf16>

  return %tensor_e : tensor<?xf16>
}


// ============================================================
// STAGE 1: 轴分组分析（编译器内部 Analysis）
//
//  对每个 Op 分析 iterator_types：
//
//   Broadcast: iterator_types=["parallel","parallel"]
//     d0 → Parallel
//     d1 → Parallel（广播轴）
//
//   Add:       iterator_types=["parallel","parallel"]
//     d0 → Parallel
//     d1 → Parallel
//
//   ReduceSum: iterator_types=["parallel","reduction"]
//     d0 → Parallel
//     d1 → Reduction（归约轴，沿N轴求和）
//
//  融合判定：
//    全局 d0 = Parallel ✅
//    全局 d1 = Reduction（与Broadcast/Add的Parallel不同）
//    → 但由于是同一计算图，可以融合为单个 Kernel
// ============================================================


// ============================================================
// STAGE 2: 融合后的 IR（Broadcast + Add + ReduceSum → 单 generic）
// ============================================================

func.func @stage2_fused_broadcast_add_reduce(
    %input_a : tensor<?xf16>,
    %input_b : tensor<?x?xf16>
) -> tensor<?xf16> {

  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %dim_m  = tensor.dim %input_a, %idx_0 : tensor<?xf16>
  %dim_n  = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

  // 融合后：Broadcast + Add + ReduceSum → 单个 linalg.generic
  %zero = arith.constant 0.0 : f16
  %empty_e = tensor.empty(%dim_m) : tensor<?xf16>
  %init_e = linalg.fill ins(%zero : f16) outs(%empty_e : tensor<?xf16>) -> tensor<?xf16>

  %tensor_e = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A：广播
      affine_map<(d0, d1) -> (d0, d1)>,  // B
      affine_map<(d0, d1) -> (d0)>       // E：归约结果
    ],
    iterator_types = ["parallel", "reduction"],
    attrs = {fusion_group = 0 : i32, fusion_segment = "broadcast_add_reduce"}
  } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
    outs(%init_e : tensor<?xf16>) {
  ^bb0(%a_val: f16, %b_val: f16, %acc: f16):
    // 内联广播加法：A[d0] + B[d0,d1]
    %sum = arith.addf %a_val, %b_val : f16
    // 累加到结果
    %new_acc = arith.addf %acc, %sum : f16
    linalg.yield %new_acc : f16
  } -> tensor<?xf16>

  return %tensor_e : tensor<?xf16>
}


// ============================================================
// STAGE 3: Tiling（TB/Tb 两级切分，沿 Parallel 轴 d0）
//
// 注意：归约轴 d1 不做切分，每个核处理完整的 N 列
// ============================================================

// Transform Dialect 调度脚本
module attributes {transform.with_named_sequence} {
  transform.named_sequence @autofuse_tiling_schedule(
      %root : !transform.any_op
  ) {
    %generic = transform.structured.match
        attributes {fusion_segment = "broadcast_add_reduce"}
        in %root : (!transform.any_op) -> !transform.any_op

    // TB级：核间切分（沿 Parallel 轴 d0）
    %tb_size = transform.param.constant 0 : i64
    %tiled_tb, %loop_tb = transform.structured.tile_using_for %generic [%tb_size, 0]
        // tile_sizes [%TB, 0]: 0 表示不切归约轴
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    transform.loop.map_to_blocks %loop_tb {block_dims = [0]}
        : (!transform.any_op) -> ()

    // Tb级：UB搬运批次（沿 Parallel 轴 d0）
    %tb_inner_size = transform.param.constant 0 : i64
    %tiled_tb_inner, %loop_tb_inner = transform.structured.tile_using_for %tiled_tb [%tb_inner_size, 0]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// Tiling后的 Affine Dialect IR
func.func @stage3_tiled_kernel(
    %input_a      : memref<?xf16>,       // [M] GM
    %input_b : memref<?x?xf16>,    // [M,N] GM
    %output : memref<?xf16>,       // [M] GM 输出
    %dim_m       : index,
    %dim_n       : index,
    %tb_size : index,
    %tb_inner_size : index,
    %core_id : index
) {
  %idx_0 = arith.constant 0 : index
  %zero_f16 = arith.constant 0.0 : f16

  // TB级：本核负责的起始偏移
  %tb_offset = arith.muli %core_id, %tb_size : index

  // Tb级循环
  affine.for %tb = 0 to %tb_size step %tb_inner_size {
    %row_start = arith.addi %tb_offset, %tb : index
    %remain     = arith.subi %tb_size, %tb : index
    %actual_tb  = arith.minsi %tb_inner_size, %remain : index

    // 分配UB Buffer
    // VECIN (9): 用于输入 A
    %buf_a   = memref.alloca(%actual_tb) : memref<?xf16, 9 : i32>
    // VECIN (9): 用于输入 B
    %buf_b   = memref.alloca(%actual_tb, %dim_n) : memref<?x?xf16, 9 : i32>
    // VECOUT (10): 用于输出 E
    %buf_e   = memref.alloca(%actual_tb) : memref<?xf16, 10 : i32>

    // DMA: A 从 GM→UB
    memref.copy
        (memref.subview %input_a[%row_start][%actual_tb][1]
         : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>),
        (memref.subview %buf_a[%idx_0][%actual_tb][1]
         : memref<?xf16, 9 : i32> to memref<?xf16, strided<[1]>, 9 : i32>)
        : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>, 9 : i32>

    // DMA: B 从 GM→UB
    memref.copy
        (memref.subview %input_b[%row_start, 0][%actual_tb, %dim_n][1, 1]
         : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>),
        (memref.subview %buf_b[%idx_0, 0][%actual_tb, %dim_n][1, 1]
         : memref<?x?xf16, 9 : i32> to memref<?x?xf16, strided<[?, 1]>, 9 : i32>)
        : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1]>, 9 : i32>

    // 初始化输出为0
    affine.for %i = 0 to %actual_tb step 1 {
      memref.store %zero_f16, %buf_e[%i] : memref<?xf16, 10 : i32>
    }

    // 核心计算：Broadcast + Add + ReduceSum
    affine.for %row = 0 to %actual_tb step 1 {
      affine.for %col = 0 to %dim_n step 1 {
        // 加载 A[row]（广播）
        %a_val = memref.load %buf_a[%row] : memref<?xf16, 9 : i32>
        // 加载 B[row, col]
        %b_val = memref.load %buf_b[%row, %col] : memref<?x?xf16, 9 : i32>
        // 加载累加器
        %acc = memref.load %buf_e[%row] : memref<?xf16, 10 : i32>
        // A + B
        %sum = arith.addf %a_val, %b_val : f16
        // 累加
        %new_acc = arith.addf %acc, %sum : f16
        // 存储
        memref.store %new_acc, %buf_e[%row] : memref<?xf16, 10 : i32>
      }
    }

    // DMA: E 从 UB→GM
    memref.copy
        (memref.subview %buf_e[%idx_0][%actual_tb][1]
         : memref<?xf16, 10 : i32> to memref<?xf16, strided<[1]>, 10 : i32>),
        (memref.subview %output[%row_start][%actual_tb][1]
         : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>)
        : memref<?xf16, strided<[1]>, 10 : i32> to memref<?xf16, strided<[1], offset: ?>>
  }

  return
}


// ============================================================
// STAGE 4: TilingData 结构 + tiling_func
// ============================================================

!TilingData = !llvm.struct<"TilingData", (
    i64,  // M
    i64,  // N
    i64,  // TB_size
    i64,  // Tb_size
    i64   // core_num
)>

func.func @tiling_func(%M : i64, %N : i64) -> !TilingData {
  // 硬件常量
  %UB_BYTES   = arith.constant 262144 : i64   // 256KB UB容量
  %CORE_NUM   = arith.constant 20 : i64       // AiCore数量
  %ELEM_BYTES = arith.constant 2 : i64        // f16 = 2字节

  // Tb_size：根据 UB 容量计算
  // 需要3个缓冲区：A切片 + B矩阵块 + 结果
  // 总元素数 = UB_BYTES / ELEM_BYTES = 131072
  %total_elems = arith.divsi %UB_BYTES, %ELEM_BYTES : i64

  // B矩阵占大部分空间：M × N
  // 简化为：Tb_size = total_elems / (N + 2)  （至少1行）
  %c2 = arith.constant 2 : i64
  %denom = arith.addi %N, %c2 : i64
  %tb_inner_size_raw = arith.divsi %total_elems, %denom : i64
  %c1 = arith.constant 1 : i64
  %tb_inner_size = arith.maxsi %tb_inner_size_raw, %c1 : i64

  // TB_size：ceil(M / CORE_NUM)，对齐Tb
  %per_core_raw = arith.addi (arith.divsi %M, %CORE_NUM : i64), %c1 : i64
  %tb_size_aligned = arith.muli
                 (arith.addi
                   (arith.divsi %per_core_raw, %tb_inner_size : i64), %c1 : i64),
                 %tb_inner_size : i64

  %core_num = arith.addi
                 (arith.divsi %M, %tb_size_aligned : i64), %c1 : i64

  %td = llvm.mlir.undef : !TilingData
  %td1 = llvm.insertvalue %M,        %td[0]  : !TilingData
  %td2 = llvm.insertvalue %N,        %td1[1] : !TilingData
  %td3 = llvm.insertvalue %tb_size_aligned,  %td2[2] : !TilingData
  %td4 = llvm.insertvalue %tb_inner_size,    %td3[3] : !TilingData
  %td5 = llvm.insertvalue %core_num, %td4[4] : !TilingData
  return %td5 : !TilingData
}


// ============================================================
// STAGE 5: Runtime Dispatch
// ============================================================

func.func @runtime_dispatch(
    %input_a   : memref<?xf16>,
    %input_b   : memref<?x?xf16>,
    %output : memref<?xf16>
) {
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index

  %dim_m = memref.dim %input_a, %idx_0 : memref<?xf16>
  %dim_n = memref.dim %input_b, %idx_1 : memref<?x?xf16>

  %M_i64 = arith.index_cast %dim_m : index to i64
  %N_i64 = arith.index_cast %dim_n : index to i64

  // 计算 TilingData
  %tiling = func.call @tiling_func(%M_i64, %N_i64)
            : (i64, i64) -> !TilingData

  // 解包
  %tb_size  = llvm.extractvalue %tiling[2] : !TilingData
  %tb_inner_size  = llvm.extractvalue %tiling[3] : !TilingData
  %core_num = llvm.extractvalue %tiling[4] : !TilingData

  // 多核下发
  %core_num_idx = arith.index_cast %core_num : i64 to index
  %tb_idx       = arith.index_cast %tb_size  : i64 to index
  %tb_inner_idx       = arith.index_cast %tb_inner_size  : i64 to index

  scf.parallel (%core_id) = (%idx_0) to (%core_num_idx) step (%idx_1) {
    func.call @stage3_tiled_kernel(
        %input_a, %input_b, %output,
        %dim_m, %dim_n,
        %tb_idx, %tb_inner_idx,
        %core_id
    ) : (memref<?xf16>, memref<?x?xf16>, memref<?xf16>,
         index, index, index, index, index) -> ()
    scf.reduce
  }

  return
}


// ============================================================
// STAGE 6: AscendC Lowering 目标代码伪表示
//
// __global__ __aicore__ void kernel_broadcast_add_reduce(
//     GM_ADDR A, GM_ADDR B, GM_ADDR E, TilingData* td
// ) {
//     int cid = GetBlockIdx();
//     int64_t row_start = cid * td->TB_size;
//
//     for (int64_t tb = 0; tb < td->TB_size; tb += td->Tb_size) {
//         int64_t abs_row = row_start + tb;
//         int64_t num_rows = min(td->Tb_size, td->TB_size - tb);
//
//         // GM→UB：A 搬运
//         DataCopy(buf_A, A + abs_row, num_rows);
//
//         // GM→UB：B 搬运（num_rows × N）
//         DataCopy(buf_B, B + abs_row * td->N, num_rows * td->N);
//
//         // 初始化输出
//         for (int i = 0; i < num_rows; i++) buf_E[i] = 0;
//
//         // 核心计算：Broadcast + Add + ReduceSum
//         for (int row = 0; row < num_rows; row++) {
//             for (int col = 0; col < td->N; col++) {
//                 half a = buf_A[row];           // 广播
//                 half b = buf_B[row * td->N + col];
//                 buf_E[row] += a + b;           // 累加
//             }
//         }
//
//         // UB→GM：E 写回
//         DataCopy(E + abs_row, buf_E, num_rows);
//     }
// }
// ============================================================
