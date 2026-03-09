// ============================================================
// STAGE 8: AscendC Kernel Code - 最终昇腾C内核代码
//
// 这是从 MLIR 降级生成的最终 AscendC C++ 内核代码
// 可以直接编译并在昇腾 AI 处理器上执行
//
// 关键组件：
//   - TPipe: 内存管道管理
//   - TQue: 异步队列管理
//   - TBuf: 片上缓冲区分配
//   - GlobalTensor/LocalTensor: 全局/本地张量抽象
//
// 执行流程：
//   1. 从 GM 读取 TilingData
//   2. 获取当前核ID (GetBlockIdx)
//   3. 计算本核的数据范围
//   4. 循环处理数据：
//      a. 数据搬运: GM → VECIN (DataCopy)
//      b. 广播: VECIN → VECCALC (Broadcast)
//      c. 计算: Add + ReduceSum
//      d. 数据写回: VECOUT → GM
//
// 内存层级：
//   - GM (__gm__): 全局内存
//   - VECIN: 向量输入缓冲区
//   - VECOUT: 向量输出缓冲区
//   - VECCALC: 向量计算缓冲区
// ============================================================

#include "kernel_operator.h"

// TilingData 结构体定义
// 包含分块参数和维度信息
struct TilingData {
  int64_t TB_M;        // 核间分块大小 (M维度)
  int64_t TB_N;        // 内层分块大小 (N维度)
  int64_t dim_arg0_0;  // M维度总大小
  int64_t dim_arg1_1;  // N维度总大小
};

// 内核函数：广播加法归约
// 使用 extern "C" 防止名称修饰
// __global__ __aicore__ 标记为昇腾内核函数
extern "C" __global__ __aicore__ void broadcast_add_reducesum(
    half* input_a,           // 输入 A [M] - GM
    half* input_b,           // 输入 B [M,N] - GM
    __gm__ TilingData* tiling_data_ptr,  // TilingData - GM
    half* output             // 输出 [M] - GM
) {
  // ---- 常量定义 ----
  constexpr uint32_t const_idx_0 = 0;    // 常量 0
  constexpr uint32_t const_idx_2 = 2;    // 常量 2 (用于字节计算)
  constexpr int32_t const_1_i32 = 1;     // 常量 1 (int32)

  // ---- 从 GM 复制 TilingData 到本地 ----
  TilingData local_tiling;
  // 逐字节复制结构体
  for (size_t i = 0; i < sizeof(local_tiling); i++) {
    auto byte = reinterpret_cast<__gm__ uint8_t*>(tiling_data_ptr)[i];
    reinterpret_cast<uint8_t*>(&local_tiling)[i] = byte;
  }

  // ---- 解包 TilingData ----
  int64_t tb_m = local_tiling.TB_M;              // 核间分块大小
  int64_t tb_n = local_tiling.TB_N;              // 内层分块大小
  int64_t dim_m = local_tiling.dim_arg0_0;       // M维度总大小
  int64_t dim_n = local_tiling.dim_arg1_1;       // N维度总大小

  // ---- 初始化 AscendC 运行时对象 ----
  // TPipe: 管理内存管道
  AscendC::TPipe pipe;
  // TQue<VECIN, 1>: 输入队列，深度为1
  AscendC::TQue<AscendC::TPosition::VECIN, 1> queue_in;
  // TQue<VECOUT, 1>: 输出队列，深度为1
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> queue_out;

  // ---- 类型转换 ----
  uint32_t tb_n_u32 = static_cast<uint32_t>(tb_n);
  uint32_t tb_m_u32 = static_cast<uint32_t>(tb_m);
  uint32_t dim_m_u32 = static_cast<uint32_t>(dim_m);
  uint32_t dim_n_u32 = static_cast<uint32_t>(dim_n);

  // ---- 分配 TBUF 缓冲区 ----
  AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf_calc_0;  // 计算缓冲区0
  AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf_calc_1;  // 计算缓冲区1
  AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf_calc_2;  // 计算缓冲区2
  AscendC::TBuf<AscendC::TPosition::VECOUT> tbuf_out;      // 输出缓冲区
  AscendC::TBuf<AscendC::TPosition::VECIN> tbuf_in;        // 输入缓冲区

  // ---- 获取当前核ID ----
  uint32_t block_idx = static_cast<uint32_t>(AscendC::GetBlockIdx());

  // ---- 计算本核的起始偏移 ----
  uint32_t block_offset = block_idx * tb_m_u32;

  // ---- 边界检查 ----
  bool is_in_bounds = block_offset < dim_m_u32;

  if (is_in_bounds) {
    // ---- 计算本核实际处理的行数 ----
    uint32_t remaining_rows = dim_m_u32 - block_offset;
    uint32_t actual_rows = (tb_m_u32 < remaining_rows) ? tb_m_u32 : remaining_rows;

    // ---- 内层循环：沿 N 维度分块 ----
    for (uint32_t inner_iv = const_idx_0; inner_iv < actual_rows; inner_iv += tb_n_u32) {
      // 计算本批次实际处理的行数
      uint32_t remaining_inner = actual_rows - inner_iv;
      uint32_t inner_size = (remaining_inner < tb_n_u32) ? remaining_inner : tb_n_u32;

      // ---- 计算缓冲区大小 ----
      uint32_t buffer_size_in = inner_size * const_idx_2;  // ×2 因为 f16 = 2 bytes

      // ---- 初始化 VECIN 缓冲区 ----
      pipe.InitBuffer(tbuf_in, buffer_size_in);

      // ---- 分配本地张量 ----
      AscendC::LocalTensor<half> local_tensor_in = queue_in.AllocTensor<half>();

      // ---- 设置全局张量 ----
      AscendC::GlobalTensor<half> global_tensor_a;

      // 计算全局偏移
      uint32_t global_offset = inner_iv + block_offset;
      int32_t global_offset_i32 = static_cast<int32_t>(global_offset);

      // 设置全局缓冲区指针
      __gm__ half* input_a_gm = reinterpret_cast<__gm__ half*>(input_a);
      global_tensor_a.SetGlobalBuffer(input_a_gm, global_offset_i32);

      // ---- 数据搬运：GM → VECIN ----
      AscendC::DataCopy(local_tensor_in, global_tensor_a, inner_size);

      // ---- 入队/出队 ----
      queue_in.EnQue(local_tensor_in);
      AscendC::LocalTensor<half> dequeued_in = queue_in.DeQue<half>();

      // ---- 初始化 VECOUT 缓冲区 ----
      pipe.InitBuffer(tbuf_out, buffer_size_in);

      // ---- 计算 B 的缓冲区大小 ----
      uint32_t b_total_size = inner_size * dim_n_u32;
      uint32_t buffer_size_b = b_total_size * const_idx_2;

      // ---- 初始化 VECCALC 缓冲区 ----
      pipe.InitBuffer(tbuf_calc_2, buffer_size_b);
      AscendC::LocalTensor<half> local_calc_2 = tbuf_calc_2.Get<half>();

      // 转换索引为 i32
      int32_t inner_size_i32 = static_cast<int32_t>(inner_size);
      int32_t dim_n_i32 = static_cast<int32_t>(dim_n_u32);

      // ---- 初始化更多 VECCALC 缓冲区 ----
      pipe.InitBuffer(tbuf_calc_1, buffer_size_b);
      AscendC::LocalTensor<half> local_calc_1 = tbuf_calc_1.Get<half>();

      // ---- 广播：将一维输入广播到二维 ----
      // 将 A [inner_size] 广播为 [inner_size, dim_n]
      AscendC::Broadcast<half, half, 2>(
          local_calc_1, dequeued_in,
          reinterpret_cast<uint64_t>(inner_size_i32),
          reinterpret_cast<uint64_t>(inner_size_i32));

      // ---- 初始化最后一个 VECCALC 缓冲区 ----
      pipe.InitBuffer(tbuf_calc_0, buffer_size_b);
      AscendC::LocalTensor<half> local_calc_0 = tbuf_calc_0.Get<half>();

      // ---- 设置 B 的全局张量 ----
      AscendC::GlobalTensor<half> global_tensor_b;

      // 计算 B 的全局偏移
      uint32_t b_offset = global_offset * dim_n_u32;
      int32_t b_offset_i32 = static_cast<int32_t>(b_offset);

      // 设置全局缓冲区指针
      __gm__ half* input_b_gm = reinterpret_cast<__gm__ half*>(input_b);
      global_tensor_b.SetGlobalBuffer(input_b_gm, b_offset_i32);

      // ---- 数据搬运：B 从 GM → VECCALC ----
      AscendC::DataCopy(local_calc_0, global_tensor_b, b_total_size);

      // ---- 向量加法：A + B ----
      AscendC::Add(local_calc_2, local_calc_1, local_calc_0, b_total_size);

      // ---- 累加 (用于归约) ----
      AscendC::Add(local_calc_2, local_calc_2, local_calc_2, b_total_size);

      // ---- 分配输出本地张量 ----
      AscendC::LocalTensor<half> local_out = queue_out.AllocTensor<half>();

      // ---- 二维归约求和 ----
      // 将 [inner_size, dim_n] 归约为 [inner_size]
      AscendC::ReduceSum<AscendC::ReduceLayout::AR>(local_out, local_calc_2);

      // ---- 入队/出队 ----
      queue_out.EnQue(local_out);
      AscendC::LocalTensor<half> dequeued_out = queue_out.DeQue<half>();

      // ---- 设置输出全局张量 ----
      AscendC::GlobalTensor<half> global_tensor_out;

      // 设置输出缓冲区指针
      __gm__ half* output_gm = reinterpret_cast<__gm__ half*>(output);
      global_tensor_out.SetGlobalBuffer(output_gm, global_offset_i32);

      // ---- 数据写回：VECOUT → GM ----
      AscendC::DataCopy(global_tensor_out, dequeued_out, inner_size);

      // ---- 释放张量 ----
      queue_out.FreeTensor(dequeued_out);
      queue_in.FreeTensor(dequeued_in);

    }  // 内层循环结束
  }  // if (is_in_bounds) 结束

  return;
}
