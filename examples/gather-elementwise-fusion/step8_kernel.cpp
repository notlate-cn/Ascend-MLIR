#include "kernel_operator.h"
#include "adv_api/broadcast/broadcast.h"
#include "adv_api/reduce/reduce.h"

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_0;
  int64_t dim_arg1_0;
  int64_t dim_arg0_1;
  int64_t dim_arg1_1;
  int64_t dim_arg2_0;
  int64_t dim_arg2_1;
};

// Computation: out[i, j] = relu(data[i, indices[j]]) + bias[j]
//   data:    (M, N) f16    v1
//   indices: (K,)   i64    v2
//   bias:    (K,)   f16    v3
//   out:     (M, K) f16    v4  (output, CANN convention)
//   workspace:             v5  (unused)
//   TilingData:            v6  (TB_M, TB_N=Tb_M, dim_arg0_0=M, dim_arg1_0=K,
//                               dim_arg0_1=N, dim_arg1_1=K, dim_arg2_0=K, dim_arg2_1=K)
//
// Tiling: block_dim = ceil(M / TB_M), each block handles TB_M rows.
//   Outer loop: Tb_M rows at a time (typically Tb_M=1 for row-by-row gather).
//   Inner loop: for each row in [0, Tb_M):
//     1. DataCopy indices[K] → VECIN (once per Tb batch)
//     2. DataCopy data[row, 0..N] → VECCALC  (one row of data, size N)
//     3. Gather: out_row[K] = data_row[indices[K]]  (uses u32 indices)
//     4. Max(out_row, out_row, 0)  → relu
//     5. DataCopy bias[K] → VECCALC
//     6. Add(out_row, out_row, bias_row, K)
//   DataCopy out[Tb_M rows] → GM output.

extern "C" __global__ __aicore__ void relu_index_select_add(
  GM_ADDR v1,
  GM_ADDR v2,
  GM_ADDR v3,
  GM_ADDR v4,
  GM_ADDR v5,
  TilingData v6
) {
  half c0_f16 = 0.0e+00;
  constexpr uint32_t c8_idx = 8;
  constexpr int32_t c1_i32 = 1;
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c0_i32 = 0;
  constexpr uint32_t c0_idx = 0;
  constexpr uint32_t c1_idx = 1;
  constexpr uint32_t c4_idx = 4;   // sizeof(uint32_t)

  int64_t v7 = v6.TB_M;    // inter-core tile (e.g. 64)
  int64_t v8 = v6.TB_N;    // intra-core tile Tb_M (e.g. 1)
  int64_t v9 = v6.dim_arg0_0;   // M
  int64_t v10 = v6.dim_arg1_0;  // K
  int64_t v11 = v6.dim_arg0_1;  // N

  AscendC::TPipe v15;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v16;   // for indices (i64)
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v17;  // for output rows

  uint32_t v18 = static_cast<uint32_t>(v8);   // Tb_M
  uint32_t v19 = static_cast<uint32_t>(v7);   // TB_M
  uint32_t v20 = static_cast<uint32_t>(v9);   // M
  uint32_t v21 = static_cast<uint32_t>(v10);  // K
  uint32_t v42 = static_cast<uint32_t>(v11);  // N

  // TBuf allocations
  AscendC::TBuf<AscendC::TPosition::VECCALC> v22;   // bias row [K] f16
  AscendC::TBuf<AscendC::TPosition::VECCALC> v23;   // zero fill [K] f16  (for relu)
  AscendC::TBuf<AscendC::TPosition::VECCALC> v24;   // data row [N] f16
  AscendC::TBuf<AscendC::TPosition::VECCALC> idx32_buf;  // indices [K] u32
  AscendC::TBuf<AscendC::TPosition::VECOUT> v25;    // output rows [Tb_M * K] f16
  AscendC::TBuf<AscendC::TPosition::VECIN> v26;     // indices [K] i64 → VECIN

  uint32_t v27 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v28 = v27 * v19;   // row start for this block
  bool v29 = v28 < v20;
  if (v29) {
    uint32_t v30 = v20 - v28;
    uint32_t v31 = ((v19 < v30) ? (v19) : (v30));  // actual rows in this block

    for (uint32_t v32 = c0_idx; v32 < v31; v32 += v18) {
      uint32_t v33 = v31 - v32;
      uint32_t v34 = ((v33 < v18) ? (v33) : (v18));  // actual Tb rows

      // ── Load indices[K] from GM → VECIN (i64, K * 8 bytes) ──────────────
      uint32_t v35 = v21 * c8_idx;
      v15.InitBuffer(v26, v35);
      v15.InitBuffer(v16, c1_i32, v35);
      AscendC::LocalTensor<int64_t> v36 = v16.AllocTensor<int64_t>();
      AscendC::GlobalTensor<int64_t> v37;
      __gm__ int64_t* v38 = reinterpret_cast<__gm__ int64_t*>(v2);
      v37.SetGlobalBuffer(v38);
      AscendC::DataCopy(v36, v37, v21);
      v16.EnQue(v36);
      AscendC::LocalTensor<int64_t> v39 = v16.DeQue<int64_t>();

      // ── Convert indices i64 → u32 in VECCALC ────────────────────────────
      // Gather requires LocalTensor<uint32_t>. Cast i64 index values to u32.
      uint32_t idx32_bytes = v21 * c4_idx;
      v15.InitBuffer(idx32_buf, idx32_bytes);
      AscendC::LocalTensor<uint32_t> idx32 = idx32_buf.Get<uint32_t>();
      for (uint32_t ii = 0; ii < v21; ii++) {
        idx32.SetValue(ii, static_cast<uint32_t>(v39.GetValue(ii)));
      }

      // ── Alloc output VECOUT [Tb_M * K] f16 ──────────────────────────────
      uint32_t v40 = v34 * v21;           // Tb_M * K elements
      uint32_t v41 = v40 * c2_idx;        // bytes
      v15.InitBuffer(v25, v41);
      v15.InitBuffer(v17, c1_i32, v41);
      int32_t v43 = static_cast<int32_t>(v21);   // K as i32 for AscendC count arg
      AscendC::LocalTensor<half> v44 = v17.AllocTensor<half>();
      uint32_t v45 = v21 * c2_idx;                // K * sizeof(half) bytes per row

      // ── Setup data global tensor (data[M,N] f16) ────────────────────────
      AscendC::GlobalTensor<half> v46;
      v46.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v1));

      // ── Alloc data row VECCALC [N] f16 ──────────────────────────────────
      uint32_t v47 = v42 * c2_idx;
      v15.InitBuffer(v24, v47);

      // ── Inner loop: process each row in [0, Tb_M) ───────────────────────
      for (uint32_t v49 = c0_idx; v49 < v34; v49 += c1_idx) {
        uint32_t v50 = v49 + v32;
        uint32_t v51 = v50 + v28;    // global row index
        uint32_t v52 = v51 * v42;    // flat offset into data[row, 0]

        // DataCopy data row [N] from GM → VECCALC
        AscendC::LocalTensor<half> v54 = v24.Get<half>();
        AscendC::GlobalTensor<half> v53;
        v53.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v1) + v52);
        AscendC::DataCopy(v54, v53, v42);

        // Gather: out_row[j] = data_row[indices[j]], for j in [0, K)
        uint32_t v55 = v49 * v45;    // byte offset into output tensor for this row
        AscendC::LocalTensor<half> v56 = v25.GetWithOffset<half>(v45, v55);
        AscendC::Gather(v56, v54, idx32, c0_i32, v43);

        // ReLU: Max(out_row, out_row, 0)
        v15.InitBuffer(v23, v45);
        AscendC::LocalTensor<half> v57 = v23.Get<half>();
        AscendC::Duplicate(v57, c0_f16, v43);
        AscendC::Max(v56, v56, v57, v43);

        // Load bias[K] from GM → VECCALC and Add
        v15.InitBuffer(v22, v45);
        AscendC::LocalTensor<half> v58 = v22.Get<half>();
        AscendC::GlobalTensor<half> v59;
        v59.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v3));
        AscendC::DataCopy(v58, v59, v21);
        AscendC::Add(v56, v56, v58, v43);
      }

      // ── Store output rows [Tb_M * K] VECOUT → GM out ────────────────────
      v17.EnQue(v44);
      AscendC::LocalTensor<half> v61 = v17.DeQue<half>();
      AscendC::GlobalTensor<half> v62;
      uint32_t v63 = v32 + v28;    // global row start for this Tb batch
      uint32_t v64 = v63 * v21;    // flat offset in output[M, K]
      v62.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v4) + v64);
      AscendC::DataCopy(v62, v61, v40);
      v17.FreeTensor(v61);
      v16.FreeTensor(v39);
    }
  }
  return;
}
