// ============================================================
// STAGE 8: AscendC Kernel Code - broadcast + add + reducesum
//
// 计算：output[m] = sum_n(input_a[m] + input_b[m, n])
//
// TilingData（6个 int64_t）：
//   TB_M, TB_N, dim_arg0_0(M), dim_arg1_1(N), dim_arg0_1(M), dim_arg1_0(N)
//
// 约束：TB_M >= 16（DataCopy half 最少 16 元素），N 为 32 的倍数且 <= 128
// ============================================================

#include "kernel_operator.h"

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_0;  // M
  int64_t dim_arg1_1;  // N
  int64_t dim_arg0_1;
  int64_t dim_arg1_0;
};

extern "C" __global__ __aicore__ void broadcast_add_reducesum(
    GM_ADDR input_a,    // [M] f16
    GM_ADDR input_b,    // [M, N] f16
    GM_ADDR output,     // [M] f16
    GM_ADDR workspace,
    TilingData tiling
) {
  uint32_t dim_m   = static_cast<uint32_t>(tiling.dim_arg0_0);
  uint32_t dim_n   = static_cast<uint32_t>(tiling.dim_arg1_1);
  uint32_t tb_m    = static_cast<uint32_t>(tiling.TB_M);  // >= 16

  uint32_t block_idx = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t row_start = block_idx * tb_m;

  if (row_start >= dim_m) return;

  uint32_t row_end  = (row_start + tb_m < dim_m) ? (row_start + tb_m) : dim_m;
  uint32_t num_rows = row_end - row_start;
  // DataCopy half 最少 16 元素（32B）
  uint32_t num_rows_aligned = ((num_rows + 15) / 16) * 16;

  uint32_t row_bytes_aligned = ((dim_n * sizeof(half) + 31) / 32) * 32;
  uint32_t out_bytes_aligned = num_rows_aligned * sizeof(half);

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1>  que_a;
  AscendC::TQue<AscendC::TPosition::VECIN, 1>  que_b;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> que_out;
  AscendC::TBuf<AscendC::TPosition::VECCALC>   tbuf_src;
  AscendC::TBuf<AscendC::TPosition::VECCALC>   tbuf_scalar;  // 32B: ReduceSum dst

  pipe.InitBuffer(que_b,       1, row_bytes_aligned);
  pipe.InitBuffer(que_a,       1, out_bytes_aligned);
  pipe.InitBuffer(que_out,     1, out_bytes_aligned);
  pipe.InitBuffer(tbuf_src,    row_bytes_aligned);
  pipe.InitBuffer(tbuf_scalar, 32);  // 16 half, only [0] used

  // 搬入 input_a[row_start : row_end]（指针直接偏移）
  __gm__ half* a_ptr = reinterpret_cast<__gm__ half*>(input_a) + row_start;
  AscendC::LocalTensor<half> local_a = que_a.AllocTensor<half>();
  AscendC::GlobalTensor<half> gtensor_a;
  gtensor_a.SetGlobalBuffer(a_ptr);
  AscendC::DataCopy(local_a, gtensor_a, num_rows_aligned);
  que_a.EnQue(local_a);

  // 分配输出 tensor（VECOUT）
  AscendC::LocalTensor<half> local_out = que_out.AllocTensor<half>();

  AscendC::LocalTensor<half> local_a_dq  = que_a.DeQue<half>();
  AscendC::LocalTensor<half> local_src   = tbuf_src.Get<half>();
  AscendC::LocalTensor<half> local_scalar = tbuf_scalar.Get<half>();

  for (uint32_t r = 0; r < num_rows; ++r) {
    // 搬入 input_b 的第 (row_start+r) 行（指针直接偏移）
    __gm__ half* b_row_ptr = reinterpret_cast<__gm__ half*>(input_b) + (row_start + r) * dim_n;
    AscendC::LocalTensor<half> local_b = que_b.AllocTensor<half>();
    AscendC::GlobalTensor<half> gtensor_b;
    gtensor_b.SetGlobalBuffer(b_row_ptr);
    AscendC::DataCopy(local_b, gtensor_b, dim_n);
    que_b.EnQue(local_b);
    AscendC::LocalTensor<half> local_b_dq = que_b.DeQue<half>();

    // local_src[n] = a[r] + b[r][n]
    half a_val = local_a_dq.GetValue(r);
    AscendC::Duplicate(local_src, a_val, dim_n);
    AscendC::Add(local_src, local_src, local_b_dq, dim_n);

    // ReduceSum: local_scalar[0] = sum(local_src[0..N])
    // dst(local_scalar) is 32B-aligned VECCALC, src(local_src) is 32B-aligned VECCALC
    AscendC::ReduceSum(local_scalar, local_src, local_scalar,
                       static_cast<int32_t>(dim_n));

    // 写入 output[r]
    local_out.SetValue(r, local_scalar.GetValue(0));

    que_b.FreeTensor(local_b_dq);
  }
  que_a.FreeTensor(local_a_dq);

  // 写回 GM output[row_start : row_end]（指针直接偏移）
  __gm__ half* out_ptr = reinterpret_cast<__gm__ half*>(output) + row_start;
  que_out.EnQue(local_out);
  AscendC::LocalTensor<half> local_out_dq = que_out.DeQue<half>();
  AscendC::GlobalTensor<half> gtensor_out;
  gtensor_out.SetGlobalBuffer(out_ptr);
  AscendC::DataCopy(gtensor_out, local_out_dq, num_rows_aligned);
  que_out.FreeTensor(local_out_dq);

  return;
}
