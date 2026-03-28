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
  int64_t v7 = v6.TB_M;
  int64_t v8 = v6.TB_N;
  int64_t v9 = v6.dim_arg0_0;
  int64_t v10 = v6.dim_arg1_0;
  int64_t v11 = v6.dim_arg0_1;
  int64_t v12 = v6.dim_arg1_1;
  int64_t v13 = v6.dim_arg2_0;
  int64_t v14 = v6.dim_arg2_1;
  AscendC::TPipe v15;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v16;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v17;
  uint32_t v18 = static_cast<uint32_t>(v8);
  uint32_t v19 = static_cast<uint32_t>(v7);
  uint32_t v20 = static_cast<uint32_t>(v9);
  uint32_t v21 = static_cast<uint32_t>(v10);
  AscendC::TBuf<AscendC::TPosition::VECCALC> v22;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v23;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v24;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v25;
  AscendC::TBuf<AscendC::TPosition::VECIN> v26;
  uint32_t v27 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v28 = v27 * v19;
  bool v29 = v28 < v20;
  if (v29) {
    uint32_t v30 = v20 - v28;
    uint32_t v31 = ((v19 < v30) ? (v19) : (v30));
    for (uint32_t v32 = c0_idx; v32 < v31; v32 += v18) {
      uint32_t v33 = v31 - v32;
      uint32_t v34 = ((v33 < v18) ? (v33) : (v18));
      uint32_t v35 = v21 * c8_idx;
      v15.InitBuffer(v26, v35);
      v15.InitBuffer(v16, c1_i32, v35);
      AscendC::LocalTensor<int64_t> v36 = v16.AllocTensor<int64_t>();
      AscendC::GlobalTensor<int64_t> v37;
      __gm__ int64_t* v38 = reinterpret_cast<__gm__ int64_t*>(v2);
      v37.SetGlobalBuffer(v38 + c0_i32);
      AscendC::DataCopy(v36, v37, v21);
      v16.EnQue(v36);
      AscendC::LocalTensor<int64_t> v39 = v16.DeQue<int64_t>();
      uint32_t v40 = v34 * v21;
      uint32_t v41 = v40 * c2_idx;
      v15.InitBuffer(v25, v41);
      v15.InitBuffer(v17, c1_i32, v41);
      uint32_t v42 = static_cast<uint32_t>(v11);
      int32_t v43 = static_cast<int32_t>(v21);
      AscendC::LocalTensor<half> v44 = v17.AllocTensor<half>();
      uint32_t v45 = v21 * c2_idx;
      AscendC::GlobalTensor<half> v46;
      v46.SetGlobalBuffer(v1);
      uint32_t v47 = v42 * c2_idx;
      v15.InitBuffer(v24, v47);
      AscendC::LocalTensor<half> v48 = v24.Get<half>();
      for (uint32_t v49 = c0_idx; v49 < v34; v49 += c1_idx) {
        uint32_t v50 = v49 + v32;
        uint32_t v51 = v50 + v28;
        uint32_t v52 = v51 * v42;
        AscendC::GlobalTensor<half> v53 = v46(v52);
        AscendC::LocalTensor<half> v54 = v24.Get<half>();
        AscendC::DataCopy(v54, v53, v42);
        uint32_t v55 = v49 * v45;
        AscendC::LocalTensor<half> v56 = v25.GetWithOffset<half>(v45, v55);
        AscendC::Gather(v56, v54, v39, c0_i32, v43);
        v15.InitBuffer(v23, v45);
        AscendC::LocalTensor<half> v57 = v23.Get<half>();
        AscendC::Duplicate(v57, c0_f16, v43);
        AscendC::Max(v56, v56, v57, v43);
        v15.InitBuffer(v22, v45);
        AscendC::LocalTensor<half> v58 = v22.Get<half>();
        AscendC::GlobalTensor<half> v59;
        __gm__ half* v60 = reinterpret_cast<__gm__ half*>(v3);
        v59.SetGlobalBuffer(v60 + c0_i32);
        AscendC::DataCopy(v58, v59, v21);
        AscendC::Add(v56, v56, v58, v43);
      }
      v17.EnQue(v44);
      AscendC::LocalTensor<half> v61 = v17.DeQue<half>();
      AscendC::GlobalTensor<half> v62;
      uint32_t v63 = v32 + v28;
      uint32_t v64 = v63 * v21;
      int32_t v65 = static_cast<int32_t>(v64);
      __gm__ half* v66 = reinterpret_cast<__gm__ half*>(v4);
      v62.SetGlobalBuffer(v66 + v65);
      AscendC::DataCopy(v62, v61, v40);
      v17.FreeTensor(v61);
      v16.FreeTensor(v39);
    }
  }
  return;
}
