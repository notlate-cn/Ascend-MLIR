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
};

extern "C" __global__ __aicore__ void relu_transpose_broadcast_add(
  GM_ADDR v1,
  GM_ADDR v2,
  GM_ADDR v3,
  GM_ADDR v4,
  TilingData v5
) {
  constexpr int32_t c0_i32 = 0;
  half c0_f16 = 0.0e+00;
  constexpr uint32_t c0_idx = 0;
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c1_i32 = 1;
  int64_t v6 = v5.TB_M;
  int64_t v7 = v5.TB_N;
  int64_t v8 = v5.dim_arg0_0;
  int64_t v9 = v5.dim_arg1_0;
  int64_t v10 = v5.dim_arg0_1;
  int64_t v11 = v5.dim_arg1_1;
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
  bool v28 = v27 < v18;
  if (v28) {
    uint32_t v29 = v18 - v27;
    uint32_t v30 = ((v16 < v29) ? (v16) : (v29));
    for (uint32_t v31 = c0_idx; v31 < v30; v31 += v15) {
      uint32_t v32 = v30 - v31;
      uint32_t v33 = ((v32 < v15) ? (v32) : (v15));
      uint32_t v34 = v17 * c2_idx;
      v12.InitBuffer(v25, v34);
      v12.InitBuffer(v13, c1_i32, v34);
      AscendC::LocalTensor<half> v35 = v13.AllocTensor<half>();
      AscendC::GlobalTensor<half> v36;
      __gm__ half* v37 = reinterpret_cast<__gm__ half*>(v1);
      v36.SetGlobalBuffer(v37 + c0_i32);
      AscendC::DataCopy(v35, v36, v17);
      v13.EnQue(v35);
      AscendC::LocalTensor<half> v38 = v13.DeQue<half>();
      uint32_t v39 = v33 * v17;
      uint32_t v40 = v39 * c2_idx;
      v12.InitBuffer(v24, v40);
      v12.InitBuffer(v14, c1_i32, v40);
      v12.InitBuffer(v23, v40);
      AscendC::LocalTensor<half> v41 = v23.Get<half>();
      int32_t v42 = static_cast<int32_t>(v33);
      int32_t v43 = static_cast<int32_t>(v17);
      v12.InitBuffer(v22, v40);
      AscendC::LocalTensor<half> v44 = v22.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v42, (uint32_t)v43};
        uint32_t _afir_ss[2] = {(uint32_t)c1_i32, (uint32_t)v43};
        AscendC::Broadcast<half, 2, 0>(v44, v38, _afir_ds, _afir_ss);
      };
      AscendC::GlobalTensor<half> v45;
      uint32_t v46 = v31 + v27;
      uint32_t v47 = static_cast<uint32_t>(v11);
      uint32_t v48 = v46 * v47;
      int32_t v49 = static_cast<int32_t>(v48);
      __gm__ half* v50 = reinterpret_cast<__gm__ half*>(v2);
      v45.SetGlobalBuffer(v50 + v49);
      v12.InitBuffer(v21, v40);
      v12.InitBuffer(v20, c1_i32, v40);
      AscendC::LocalTensor<half> v51 = v20.AllocTensor<half>();
      AscendC::DataCopy(v51, v45, v39);
      v20.EnQue(v51);
      AscendC::LocalTensor<half> v52 = v20.DeQue<half>();
      v12.InitBuffer(v19, v40);
      AscendC::LocalTensor<half> v53 = v19.Get<half>();
      AscendC::Duplicate(v53, c0_f16, v39);
      AscendC::Max(v41, v44, v53, v39);
      AscendC::Add(v41, v41, v52, v39);
      v14.EnQue(v41);
      AscendC::LocalTensor<half> v54 = v14.DeQue<half>();
      AscendC::GlobalTensor<half> v55;
      uint32_t v56 = v46 * v17;
      int32_t v57 = static_cast<int32_t>(v56);
      __gm__ half* v58 = reinterpret_cast<__gm__ half*>(v3);
      v55.SetGlobalBuffer(v58 + v57);
      AscendC::DataCopy(v55, v54, v39);
      v14.FreeTensor(v54);
      v13.FreeTensor(v38);
    }
  }
  return;
}
