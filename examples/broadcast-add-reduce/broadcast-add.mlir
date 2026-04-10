// ============================================================
// AutoFuse 完整MLIR表示 - 广播加法示例
// 计算图：A[M] → Broadcast → C[M,N]
//         C[M,N] + B[M,N] → D[M,N] (逐元素加法)
//         Store D[M,N]
//
// 本示例展示了一个完整的编译流程：
//   Broadcast + Add 融合 → 轴合并 → Tiling → AscendC代码生成
//
// 与 broadcast-add-reduce.mlir 的区别：
//   - 本例只有 Parallel 轴，没有 Reduce 轴
//   - 可以全局融合为单个 Kernel
//   - 轴可以合并为 1D flat 空间，简化 Tiling
//
// 编译流程：
//   STAGE 0: High-Level IR (tensor/linalg，完全符号化)
//   STAGE 1: 轴分组分析 (所有轴都是 Parallel)
//   STAGE 2: 融合判定 (可全局融合)
//   STAGE 3: 轴合并 [M,N] → [MN]
//   STAGE 4: Tiling (TB/Tb/t 三级)
//   STAGE 5: TilingData 结构 + tiling_func
//   STAGE 6: Runtime Dispatch (多核下发)
//   STAGE 7: AscendC Lowering
// ============================================================


// ============================================================
// STAGE 0: 输入的 High-Level IR（完全符号化，M/N 动态Shape）
// ============================================================
//
//   A : tensor<?xf16>      shape=[M]      Load1
//   B : tensor<?x?xf16>    shape=[M,N]    Load2
//   D : tensor<?x?xf16>    shape=[M,N]    输出
//
//   Op1: Broadcast  A[M] → C[M,N]
//   Op2: Add        D[M,N] = C[M,N] + B[M,N]

func.func @stage0_input(
    %input_a : tensor<?xf16>,       // shape [M]
    %input_b : tensor<?x?xf16>      // shape [M, N]
) -> tensor<?x?xf16> {

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

  return %tensor_d : tensor<?x?xf16>
}


// ============================================================
// STAGE 1: 轴分组分析（编译器内部 Analysis）
//
//  对每个 Op 分析 iterator_types：
//
//   Broadcast: iterator_types=["parallel","parallel"]
//     d0 → Parallel
//     d1 → Parallel（广播轴，输出侧parallel）
//
//   Add:       iterator_types=["parallel","parallel"]
//     d0 → Parallel
//     d1 → Parallel
//
//  融合判定：
//    全局 d0 = Parallel ✅
//    全局 d1 = Parallel ✅
//    → 所有轴分组一致 → 可全局融合！
// ============================================================


// ============================================================
// STAGE 2: 融合后的 IR（Broadcast + Add → 单 generic）
// ============================================================

func.func @stage2_fused_broadcast_add(
    %input_a : tensor<?xf16>,
    %input_b : tensor<?x?xf16>
) -> tensor<?x?xf16> {

  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %dim_m  = tensor.dim %input_a, %idx_0 : tensor<?xf16>
  %dim_n  = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>

  // 融合后：Broadcast + Add → 单个 linalg.generic
  %empty_d = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
  %tensor_d = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,      // A：广播
      affine_map<(d0, d1) -> (d0, d1)>,  // B
      affine_map<(d0, d1) -> (d0, d1)>   // D
    ],
    iterator_types = ["parallel", "parallel"],
    attrs = {fusion_group = 0 : i32, fusion_segment = "broadcast_add"}
  } ins(%input_a, %input_b : tensor<?xf16>, tensor<?x?xf16>)
    outs(%empty_d : tensor<?x?xf16>) {
  ^bb0(%a_val: f16, %b_val: f16, %out: f16):
    %sum = arith.addf %a_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?x?xf16>

  return %tensor_d : tensor<?x?xf16>
}


// ============================================================
// STAGE 3: 轴合并（Axis Merging）
//          [d0=M, d1=N] 两个Parallel轴 → 合并为 [d_flat=MN]
// ============================================================

func.func @stage3_axis_merged(
    %input_a : tensor<?xf16>,
    %input_b : tensor<?x?xf16>
) -> tensor<?x?xf16> {

  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %dim_m  = tensor.dim %input_a, %idx_0 : tensor<?xf16>
  %dim_n  = tensor.dim %input_b, %idx_1 : tensor<?x?xf16>
  %dim_mn = arith.muli %dim_m, %dim_n : index

  // B: [M,N] → [MN]，collapse两个Parallel轴
  %b_flat = tensor.collapse_shape %input_b [[0, 1]]
            : tensor<?x?xf16> into tensor<?xf16>

  // 在合并后的1D空间上计算
  %empty_flat = tensor.empty(%dim_mn) : tensor<?xf16>
  %d_flat = linalg.generic {
    indexing_maps = [
      affine_map<(d_flat)[s0] -> (d_flat floordiv s0)>,  // A[row]，s0=N
      affine_map<(d_flat) -> (d_flat)>,                   // B_flat
      affine_map<(d_flat) -> (d_flat)>                    // D_flat
    ],
    iterator_types = ["parallel"],
    attrs = {axis_merged = true, segment = "broadcast_add",
             original_axes = "M,N", merged_symbol = "MN"}
  } ins(%input_a, %b_flat : tensor<?xf16>, tensor<?xf16>)
    outs(%empty_flat : tensor<?xf16>) {
  ^bb0(%a_val: f16, %b_val: f16, %out: f16):
    %sum = arith.addf %a_val, %b_val : f16
    linalg.yield %sum : f16
  } -> tensor<?xf16>

  // 结果 reshape 回 [M,N]
  %d_2d = tensor.expand_shape %d_flat [[0, 1]]
          output_shape [%dim_m, %dim_n]
          : tensor<?xf16> into tensor<?x?xf16>

  return %d_2d : tensor<?x?xf16>
}


// ============================================================
// STAGE 4: Tiling（TB/Tb/t 三级切分）
// ============================================================

// Transform Dialect 调度脚本
module attributes {transform.with_named_sequence} {
  transform.named_sequence @autofuse_tiling_schedule(
      %root : !transform.any_op
  ) {
    %generic = transform.structured.match
        attributes {axis_merged = true}
        in %root : (!transform.any_op) -> !transform.any_op

    // TB级：核间切分
    %tb_size = transform.param.constant 0 : i64
    %tiled_tb, %loop_tb = transform.structured.tile_using_for %generic [%tb_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    transform.loop.map_to_blocks %loop_tb {block_dims = [0]}
        : (!transform.any_op) -> ()

    // Tb级：UB搬运批次
    %tb_inner_size = transform.param.constant 0 : i64
    %tiled_tb_inner, %loop_tb_inner = transform.structured.tile_using_for %tiled_tb [%tb_inner_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)

    // t级：向量化粒度（128个f16）
    %t_size = transform.param.constant 128 : i64
    %tiled_t, %loop_t = transform.structured.tile_using_for %tiled_tb_inner [%t_size]
        : (!transform.any_op, !transform.param<i64>)
       -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled_t : !transform.any_op

    transform.yield
  }
}

// Tiling后的 Affine Dialect IR
func.func @stage4_tiled_kernel(
    %input_a      : memref<?xf16>,       // [M] GM
    %b_flat : memref<?xf16>,       // [MN] GM
    %d_flat : memref<?xf16>,       // [MN] GM 输出
    %dim_mn      : index,
    %dim_n       : index,
    %tb_size : index,
    %tb_inner_size : index,
    %t_size  : index,
    %core_id : index
) {
  %idx_0 = arith.constant 0 : index

  // TB级：本核负责的起始偏移
  %tb_offset = arith.muli %core_id, %tb_size : index

  // Tb级循环
  affine.for %tb = 0 to %tb_size step %tb_inner_size {
    %flat_start = arith.addi %tb_offset, %tb : index
    %remain     = arith.subi %tb_size, %tb : index
    %actual_tb  = arith.minsi %tb_inner_size, %remain : index

    // 分配UB Buffer
    %buf_a   = memref.alloca(%tb_inner_size) : memref<?xf16, 9 : i32>
    %buf_b   = memref.alloca(%tb_inner_size) : memref<?xf16, 9 : i32>
    %buf_d   = memref.alloca(%tb_inner_size) : memref<?xf16, 10 : i32>

    // DMA: B_flat 连续段 GM→UB
    memref.copy
        (memref.subview %b_flat[%flat_start][%actual_tb][1]
         : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>),
        (memref.subview %buf_b[%idx_0][%actual_tb][1]
         : memref<?xf16, 9 : i32> to memref<?xf16, strided<[1]>, 9 : i32>)
        : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>, 9 : i32>

    // Broadcast DMA: A GM→UB
    affine.for %i = 0 to %actual_tb step 1 {
      %abs_idx = arith.addi %flat_start, %i : index
      %row     = arith.divui %abs_idx, %dim_n : index
      %a_elem  = memref.load %input_a[%row] : memref<?xf16>
      memref.store %a_elem, %buf_a[%i] : memref<?xf16, 9 : i32>
    }

    // t级：向量化 Add
    affine.for %t = 0 to %actual_tb step %t_size {
      %actual_t = arith.minsi %t_size, (arith.subi %actual_tb, %t : index) : index
      %vec_a  = vector.load %buf_a[%t]  : memref<?xf16, 9 : i32>, vector<128xf16>
      %vec_b  = vector.load %buf_b[%t]  : memref<?xf16, 9 : i32>, vector<128xf16>
      %vec_d  = arith.addf %vec_a, %vec_b : vector<128xf16>
      vector.store %vec_d, %buf_d[%t] : memref<?xf16, 10 : i32>, vector<128xf16>
    }

    // DMA: D UB→GM
    memref.copy
        (memref.subview %buf_d[%idx_0][%actual_tb][1]
         : memref<?xf16, 10 : i32> to memref<?xf16, strided<[1]>, 10 : i32>),
        (memref.subview %d_flat[%flat_start][%actual_tb][1]
         : memref<?xf16> to memref<?xf16, strided<[1], offset: ?>>)
        : memref<?xf16, strided<[1]>, 10 : i32> to memref<?xf16, strided<[1], offset: ?>>
  }

  return
}


// ============================================================
// STAGE 5: TilingData 结构 + tiling_func
// ============================================================

!TilingData = !llvm.struct<"TilingData", (
    i64,  // M
    i64,  // N
    i64,  // MN = M*N
    i64,  // TB_size
    i64,  // Tb_size
    i64,  // t_size
    i64   // core_num
)>

func.func @tiling_func(%M : i64, %N : i64) -> !TilingData {
  // 硬件常量
  %UB_BYTES   = arith.constant 262144 : i64
  %CORE_NUM   = arith.constant 20 : i64
  %ELEM_BYTES = arith.constant 2 : i64
  %VEC_WIDTH  = arith.constant 128 : i64

  %MN    = arith.muli %M, %N : i64
  %t_size = arith.constant 128 : i64

  // Tb_size：UB三等分，对齐128
  %total_elems = arith.divsi %UB_BYTES, %ELEM_BYTES : i64
  %buf_elems   = arith.divsi %total_elems,
                              (arith.constant 3 : i64) : i64
  %tb_inner_size = arith.muli
                   (arith.divsi %buf_elems, %t_size : i64),
                   %t_size : i64

  // TB_size：ceil(MN / CORE_NUM)，对齐Tb
  %c1        = arith.constant 1 : i64
  %per_core  = arith.addi (arith.divsi %MN, %CORE_NUM : i64), %c1 : i64
  %tb_size   = arith.muli
                 (arith.addi
                   (arith.divsi %per_core, %tb_inner_size : i64), %c1 : i64),
                 %tb_inner_size : i64

  %core_num  = arith.addi
                 (arith.divsi %MN, %tb_size : i64), %c1 : i64

  %td = llvm.mlir.undef : !TilingData
  %td1 = llvm.insertvalue %M,        %td[0]  : !TilingData
  %td2 = llvm.insertvalue %N,        %td1[1] : !TilingData
  %td3 = llvm.insertvalue %MN,       %td2[2] : !TilingData
  %td4 = llvm.insertvalue %tb_size,  %td3[3] : !TilingData
  %td5 = llvm.insertvalue %tb_inner_size,  %td4[4] : !TilingData
  %td6 = llvm.insertvalue %t_size,   %td5[5] : !TilingData
  %td7 = llvm.insertvalue %core_num, %td6[6] : !TilingData
  return %td7 : !TilingData
}


// ============================================================
// STAGE 6: Runtime Dispatch
// ============================================================

func.func @runtime_dispatch(
    %input_a   : memref<?xf16>,
    %input_b   : memref<?x?xf16>,
    %output : memref<?x?xf16>
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
  %tb_size  = llvm.extractvalue %tiling[3] : !TilingData
  %tb_inner_size  = llvm.extractvalue %tiling[4] : !TilingData
  %t_size   = llvm.extractvalue %tiling[5] : !TilingData
  %core_num = llvm.extractvalue %tiling[6] : !TilingData

  // B和输出展平
  %dim_mn = arith.muli %dim_m, %dim_n : index
  %b_flat = memref.collapse_shape %input_b [[0,1]]
            : memref<?x?xf16> into memref<?xf16>
  %d_flat = memref.collapse_shape %output [[0,1]]
            : memref<?x?xf16> into memref<?xf16>

  // 多核下发
  %core_num_idx = arith.index_cast %core_num : i64 to index
  %tb_idx       = arith.index_cast %tb_size  : i64 to index
  %tb_inner_idx       = arith.index_cast %tb_inner_size  : i64 to index
  %t_idx        = arith.index_cast %t_size   : i64 to index
  %mn_idx       = arith.index_cast
                    (llvm.extractvalue %tiling[2] : !TilingData)
                    : i64 to index

  scf.parallel (%core_id) = (%idx_0) to (%core_num_idx) step (%idx_1) {
    func.call @stage4_tiled_kernel(
        %input_a, %b_flat, %d_flat,
        %mn_idx, %dim_n,
        %tb_idx, %tb_inner_idx, %t_idx,
        %core_id
    ) : (memref<?xf16>, memref<?xf16>, memref<?xf16>,
         index, index, index, index, index, index) -> ()
    scf.reduce
  }

  return
}


// ============================================================
// STAGE 7: AscendC Lowering 目标代码伪表示
//
// __global__ __aicore__ void kernel_broadcast_add(
//     GM_ADDR A, GM_ADDR B, GM_ADDR D, TilingData* td
// ) {
//     int cid = GetBlockIdx();
//     int64_t flat_start = cid * td->TB_size;
//
//     for (int64_t tb = 0; tb < td->TB_size; tb += td->Tb_size) {
//         int64_t abs = flat_start + tb;
//         int64_t len = min(td->Tb_size, td->TB_size - tb);
//
//         // GM→UB：B 连续搬运
//         DataCopy(buf_B, B + abs, len);
//
//         // GM→UB：A 广播
//         for (int i = 0; i < len; i++)
//             buf_A[i] = A[(abs + i) / td->N];
//
//         // 向量加法
//         for (int64_t t = 0; t < len; t += 128)
//             Add(buf_D + t, buf_A + t, buf_B + t, 128);
//
//         // UB→GM：D 写回
//         DataCopy(D + abs, buf_D, len);
//     }
// }
// ============================================================
