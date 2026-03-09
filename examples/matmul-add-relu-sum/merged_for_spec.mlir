// ============================================================
// merged_for_spec.mlir — 用于Spec生成的合并IR
//
// 本文件将编译流程中的关键阶段合并，用于生成完整的Spec文档
//
// 包含阶段：
//   - Stage 0: 高层IR (fc_add_relu.mlir)
//   - Stage 1: Tiling后的IR
//   - Stage 2: Bufferized IR
//   - Stage 3: Buffer Placement IR
//   - Stage 4: AscendC Lowering
//
// 计算图: output = ReLU(matmul(input_a, input_b) + bias)
// ============================================================

// ============================================================
// PART 1: 高层IR (Stage 0)
// ============================================================

// 主函数: 全连接层 + ReLU
func.func @fc_relu_stage0(
    %input_a: tensor<?x?xf32>,   // 输入A [M, K]
    %input_b: tensor<?x?xf32>,   // 输入B [K, N] (权重)
    %bias: tensor<?x?xf32>,       // 偏置 [M, N]
    %output: tensor<?x?xf32>     // 输出 [M, N]
) -> tensor<?x?xf32> {

  // 操作1: 矩阵乘法
  %matmul_result = linalg.matmul
    ins(%input_a, %input_b : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // 操作2: 加偏置
  %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul_result, %bias : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // 操作3: ReLU
  %zero_f32 = arith.constant 0.0 : f32
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %dim_m = tensor.dim %biased, %idx_0 : tensor<?x?xf32>
  %dim_n = tensor.dim %biased, %idx_1 : tensor<?x?xf32>
  %empty_zero = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
  %zero_tensor = linalg.fill
    ins(%zero_f32 : f32)
    outs(%empty_zero : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  %relu_result = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%biased, %zero_tensor : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%output : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  return %relu_result : tensor<?x?xf32>
}

// ============================================================
// PART 2: Tiling后的IR (Stage 1)
// ============================================================

#tile_guard = affine_map<(d0)[upper, tile] -> (-d0 + upper, tile)>

func.func @fc_relu_stage1(
    %input_a: tensor<?x?xf32>,
    %input_b: tensor<?x?xf32>,
    %bias: tensor<?x?xf32>,
    %output_init: tensor<?x?xf32>,
    %tile_m_outer: i64, %tile_n_outer: i64,
    %tile_m_inner: i64, %tile_n_inner: i64,
    %tile_k: i64
) -> tensor<?x?xf32> {

  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %zero_f32 = arith.constant 0.000000e+00 : f32

  // 类型转换
  %tb_m = arith.index_cast %tile_m_outer : i64 to index
  %tb_n = arith.index_cast %tile_n_outer : i64 to index
  %tb_m_inner = arith.index_cast %tile_m_inner : i64 to index
  %tb_n_inner = arith.index_cast %tile_n_inner : i64 to index
  %t_k = arith.index_cast %tile_k : i64 to index

  %dim_m = tensor.dim %output_init, %idx_0 : tensor<?x?xf32>
  %dim_n = tensor.dim %output_init, %idx_1 : tensor<?x?xf32>

  // 三级Tiling结构
  %result = scf.for %m_outer = %idx_0 to %dim_m step %tb_m
      iter_args(%acc_outer = %output_init) -> (tensor<?x?xf32>) {

    %result_n = scf.for %n_outer = %idx_0 to %dim_n step %tb_n
        iter_args(%acc_n = %acc_outer) -> (tensor<?x?xf32>) {

      %m_size = affine.min #tile_guard(%m_outer)[%dim_m, %tb_m]
      %n_size = affine.min #tile_guard(%n_outer)[%dim_n, %tb_n]
      %dim_k = tensor.dim %input_a, %idx_1 : tensor<?x?xf32>

      // 提取tile
      %a_tile = tensor.extract_slice %input_a[%m_outer, 0][%m_size, %dim_k][1, 1]
                : tensor<?x?xf32> to tensor<?x?xf32>
      %b_tile = tensor.extract_slice %input_b[0, %n_outer][%dim_k, %n_size][1, 1]
                : tensor<?x?xf32> to tensor<?x?xf32>

      // 内层Tiling和计算...
      // (省略详细实现，见exp_output_step1.mlir)

      scf.yield %acc_n : tensor<?x?xf32>
    } {ascendc.parallel = true}

    scf.yield %result_n : tensor<?x?xf32>
  } {ascendc.parallel = true}

  return %result : tensor<?x?xf32>
}

// ============================================================
// PART 3: Bufferized IR (Stage 2)
// ============================================================

func.func @fc_relu_stage2(
    %input_a: memref<?x?xf32>,
    %input_b: memref<?x?xf32>,
    %bias: memref<?x?xf32>,
    %output: memref<?x?xf32>,
    %tile_params: ...
) -> memref<?x?xf32> {
  // Tensor -> Memref转换
  // 使用memref.subview进行切片
  // 使用memref.alloc分配临时缓冲区
  // (详细实现见exp_output_step2.mlir)
  return %output : memref<?x?xf32>
}

// ============================================================
// PART 4: Buffer Placement IR (Stage 3)
// ============================================================

func.func @fc_relu_stage3(
    %input_a: memref<?x?xf32>,
    %input_b: memref<?x?xf32>,
    %bias: memref<?x?xf32>,
    %output: memref<?x?xf32>,
    %tile_params: ...
) -> memref<?x?xf32> {
  // 使用memory_space指定内存位置:
  //   - A1 (1): L1缓冲区
  //   - A2 (2): L0A缓冲区
  //   - B1 (3): L1缓冲区
  //   - B2 (4): L0B缓冲区
  //   - CO1 (7): L0C缓冲区
  //   - VECIN (9): UB输入
  //   - VECOUT (10): UB输出
  // (详细实现见exp_output_step3.mlir)
  return %output : memref<?x?xf32>
}

// ============================================================
// PART 5: AscendC Lowering (Stage 4)
// ============================================================

// AscendC内核函数声明
func.func @fc_relu_ascendc(
    %input_a: memref<?x?xf32, 22>,   // GM (Global Memory)
    %input_b: memref<?x?xf32, 22>,
    %bias: memref<?x?xf32, 22>,
    %output: memref<?x?xf32, 22>,
    %tiling_data: memref<?xi64, 22>
) attributes {ascendc.aicore, ascendc.global} {
  // 使用AscendC方言:
  //   - ascendc.pipe: 内存管道
  //   - ascendc.queue: 异步队列
  //   - ascendc.tbuf: 片上缓冲区
  //   - ascendc.data_copy: 数据搬运
  //   - ascendc.matmul: 矩阵乘法
  //   - ascendc.add_l2: 向量加法
  //   - ascendc.max_l2: ReLU
  // (详细实现见output_step4_lowering_to_asc.mlir)
  return
}
