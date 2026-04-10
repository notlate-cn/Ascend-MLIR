#include "kernel_operator.h"
#include "adv_api/broadcast/broadcast.h"
#include "adv_api/reduce/reduce.h"

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_0;
  int64_t dim_arg1_1;
  int64_t dim_arg0_1;
  int64_t dim_arg1_0;
};

extern "C" __global__ __aicore__ void broadcast_add_reducesum(
  GM_ADDR v1,
  GM_ADDR v2,
  GM_ADDR v3,
  GM_ADDR v4,
  TilingData v5
) {
  half c0_f16 = 0.0e+00;
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c1_i32 = 1;
  constexpr uint32_t c0_idx = 0;
  int64_t v6 = v5.TB_M;
  int64_t v7 = v5.TB_N;
  int64_t v8 = v5.dim_arg0_0;
  int64_t v9 = v5.dim_arg1_1;
  int64_t v10 = v5.dim_arg0_1;
  int64_t v11 = v5.dim_arg1_0;
  AscendC::TPipe v12;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v13;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v14;
  uint32_t v15 = static_cast<uint32_t>(v7);
  uint32_t v16 = static_cast<uint32_t>(v6);
  uint32_t v17 = static_cast<uint32_t>(v8);
  uint32_t v18 = static_cast<uint32_t>(v9);
  AscendC::TBuf<AscendC::TPosition::VECCALC> v19;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v20;
  AscendC::TBuf<AscendC::TPosition::VECIN> v21;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v22;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v23;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v24;
  AscendC::TBuf<AscendC::TPosition::VECIN> v25;
  uint32_t v26 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v27 = v26 * v16;
  bool v28 = v27 < v17;
  if (v28) {
    uint32_t v29 = v17 - v27;
    uint32_t v30 = ((v16 < v29) ? (v16) : (v29));
    for (uint32_t v31 = c0_idx; v31 < v30; v31 += v15) {
      uint32_t v32 = v30 - v31;
      uint32_t v33 = ((v32 < v15) ? (v32) : (v15));
      uint32_t v34 = v33 * c2_idx;
      v12.InitBuffer(v25, v34);
      v12.InitBuffer(v13, c1_i32, v34);
      AscendC::LocalTensor<half> v35 = v13.AllocTensor<half>();
      AscendC::GlobalTensor<half> v36;
      uint32_t v37 = v31 + v27;
      int32_t v38 = static_cast<int32_t>(v37);
      __gm__ half* v39 = reinterpret_cast<__gm__ half*>(v1);
      v36.SetGlobalBuffer(v39 + v38);
      AscendC::DataCopy(v35, v36, v33);
      v13.EnQue(v35);
      AscendC::LocalTensor<half> v40 = v13.DeQue<half>();
      v12.InitBuffer(v24, v34);
      v12.InitBuffer(v14, c1_i32, v34);
      uint32_t v41 = v33 * v18;
      uint32_t v42 = v41 * c2_idx;
      v12.InitBuffer(v23, v42);
      AscendC::LocalTensor<half> v43 = v23.Get<half>();
      AscendC::Duplicate(v43, c0_f16, v41);
      int32_t v44 = static_cast<int32_t>(v33);
      int32_t v45 = static_cast<int32_t>(v18);
      v12.InitBuffer(v22, v42);
      AscendC::LocalTensor<half> v46 = v22.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v44, (uint32_t)v45};
        uint32_t _afir_ss[2] = {(uint32_t)v44, (uint32_t)c1_i32};
        AscendC::Broadcast<half, 2, 1>(v46, v40, _afir_ds, _afir_ss);
      };
      AscendC::GlobalTensor<half> v47;
      uint32_t v48 = v37 * v18;
      int32_t v49 = static_cast<int32_t>(v48);
      __gm__ half* v50 = reinterpret_cast<__gm__ half*>(v2);
      v47.SetGlobalBuffer(v50 + v49);
      v12.InitBuffer(v21, v42);
      v12.InitBuffer(v20, c1_i32, v42);
      AscendC::LocalTensor<half> v51 = v20.AllocTensor<half>();
      AscendC::DataCopy(v51, v47, v41);
      v20.EnQue(v51);
      AscendC::LocalTensor<half> v52 = v20.DeQue<half>();
      v12.InitBuffer(v19, v42);
      AscendC::LocalTensor<half> v53 = v19.Get<half>();
      AscendC::Add(v53, v46, v52, v41);
      AscendC::Add(v43, v43, v53, v41);
      AscendC::LocalTensor<half> v54 = v14.AllocTensor<half>();
      {
        uint32_t _afir_rows = (uint32_t)(v34 / sizeof(half));
        uint32_t _afir_cols = (uint32_t)(v42 / v34);
        AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_dst;
        AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_ws;
        v12.InitBuffer(_afir_tbuf_dst, 32);
        v12.InitBuffer(_afir_tbuf_ws, 32);
        AscendC::LocalTensor<half> _afir_scalar = _afir_tbuf_dst.Get<half>();
        AscendC::LocalTensor<half> _afir_ws = _afir_tbuf_ws.Get<half>();
        for (uint32_t _afir_r = 0; _afir_r < _afir_rows; _afir_r++) {
          AscendC::ReduceSum<half>(_afir_scalar, v43[_afir_r * _afir_cols],
                                  _afir_ws, (int32_t)_afir_cols);
          v54.SetValue(_afir_r, _afir_scalar.GetValue(0));
        }
      };
      v14.EnQue(v54);
      AscendC::LocalTensor<half> v55 = v14.DeQue<half>();
      AscendC::GlobalTensor<half> v56;
      __gm__ half* v57 = reinterpret_cast<__gm__ half*>(v3);
      v56.SetGlobalBuffer(v57 + v38);
      AscendC::DataCopy(v56, v55, v33);
      v14.FreeTensor(v55);
      v13.FreeTensor(v40);
    }
  }
  return;
}
