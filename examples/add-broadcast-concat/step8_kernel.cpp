#include "kernel_operator.h"
#include "adv_api/broadcast/broadcast.h"
#include "adv_api/reduce/reduce.h"

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_0;
  int64_t dim_arg1_1;
  int64_t dim_arg2_0;
  int64_t dim_arg3_1;
  int64_t dim_arg0_1;
  int64_t dim_arg1_0;
  int64_t dim_arg2_1;
  int64_t dim_arg3_0;
};

extern "C" __global__ __aicore__ void ewop_broadcast_concat(
  GM_ADDR v1,
  GM_ADDR v2,
  GM_ADDR v3,
  GM_ADDR v4,
  GM_ADDR v5,
  GM_ADDR v6,
  TilingData v7
) {
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c1_i32 = 1;
  constexpr uint32_t c0_idx = 0;
  int64_t v8 = v7.TB_M;
  int64_t v9 = v7.TB_N;
  int64_t v10 = v7.dim_arg0_0;
  int64_t v11 = v7.dim_arg1_1;
  int64_t v12 = v7.dim_arg2_0;
  int64_t v13 = v7.dim_arg3_1;
  int64_t v14 = v7.dim_arg0_1;
  int64_t v15 = v7.dim_arg1_0;
  int64_t v16 = v7.dim_arg2_1;
  int64_t v17 = v7.dim_arg3_0;
  AscendC::TPipe v18;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v19;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v20;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v21;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v22;
  uint32_t v23 = static_cast<uint32_t>(v9);
  uint32_t v24 = static_cast<uint32_t>(v8);
  uint32_t v25 = static_cast<uint32_t>(v10);
  uint32_t v26 = static_cast<uint32_t>(v11);
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v27;
  AscendC::TBuf<AscendC::TPosition::VECIN> v28;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v29;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v30;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v31;
  AscendC::TBuf<AscendC::TPosition::VECIN> v32;
  uint32_t v33 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v34 = v33 * v24;
  bool v35 = v34 < v25;
  if (v35) {
    uint32_t v36 = v25 - v34;
    uint32_t v37 = ((v24 < v36) ? (v24) : (v36));
    for (uint32_t v38 = c0_idx; v38 < v37; v38 += v23) {
      uint32_t v39 = v37 - v38;
      uint32_t v40 = ((v39 < v23) ? (v39) : (v23));
      uint32_t v41 = v40 * c2_idx;
      v18.InitBuffer(v32, v41);
      v18.InitBuffer(v19, c1_i32, v41);
      AscendC::LocalTensor<half> v42 = v19.AllocTensor<half>();
      AscendC::GlobalTensor<half> v43;
      uint32_t v44 = v38 + v34;
      int32_t v45 = static_cast<int32_t>(v44);
      __gm__ half* v46 = reinterpret_cast<__gm__ half*>(v1);
      v43.SetGlobalBuffer(v46 + v45);
      AscendC::DataCopy(v42, v43, v40);
      v19.EnQue(v42);
      AscendC::LocalTensor<half> v47 = v19.DeQue<half>();
      uint32_t v48 = v40 * v26;
      uint32_t v49 = v48 * c2_idx;
      v18.InitBuffer(v31, v49);
      v18.InitBuffer(v20, c1_i32, v49);
      v18.InitBuffer(v30, v49);
      AscendC::LocalTensor<half> v50 = v30.Get<half>();
      int32_t v51 = static_cast<int32_t>(v40);
      int32_t v52 = static_cast<int32_t>(v26);
      v18.InitBuffer(v29, v49);
      AscendC::LocalTensor<half> v53 = v29.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v51, (uint32_t)v52};
        uint32_t _afir_ss[2] = {(uint32_t)v51, (uint32_t)c1_i32};
        AscendC::Broadcast<half, 2, 1>(v53, v47, _afir_ds, _afir_ss);
      };
      AscendC::GlobalTensor<half> v54;
      uint32_t v55 = v44 * v26;
      int32_t v56 = static_cast<int32_t>(v55);
      __gm__ half* v57 = reinterpret_cast<__gm__ half*>(v2);
      v54.SetGlobalBuffer(v57 + v56);
      v18.InitBuffer(v28, v49);
      v18.InitBuffer(v27, c1_i32, v49);
      AscendC::LocalTensor<half> v58 = v27.AllocTensor<half>();
      AscendC::DataCopy(v58, v54, v48);
      v27.EnQue(v58);
      AscendC::LocalTensor<half> v59 = v27.DeQue<half>();
      AscendC::Add(v50, v53, v59, v48);
      v20.EnQue(v50);
      AscendC::LocalTensor<half> v60 = v20.DeQue<half>();
      AscendC::GlobalTensor<half> v61;
      __gm__ half* v62 = reinterpret_cast<__gm__ half*>(v5);
      v61.SetGlobalBuffer(v62 + v56);
      AscendC::DataCopy(v61, v60, v48);
      v20.FreeTensor(v60);
      v19.FreeTensor(v47);
    }
  }
  uint32_t v63 = static_cast<uint32_t>(v12);
  uint32_t v64 = static_cast<uint32_t>(v13);
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v65;
  AscendC::TBuf<AscendC::TPosition::VECIN> v66;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v67;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v68;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v69;
  AscendC::TBuf<AscendC::TPosition::VECIN> v70;
  bool v71 = v34 < v63;
  if (v71) {
    uint32_t v72 = v63 - v34;
    uint32_t v73 = ((v24 < v72) ? (v24) : (v72));
    for (uint32_t v74 = c0_idx; v74 < v73; v74 += v23) {
      uint32_t v75 = v73 - v74;
      uint32_t v76 = ((v75 < v23) ? (v75) : (v23));
      uint32_t v77 = v76 * c2_idx;
      v18.InitBuffer(v70, v77);
      v18.InitBuffer(v21, c1_i32, v77);
      AscendC::LocalTensor<half> v78 = v21.AllocTensor<half>();
      AscendC::GlobalTensor<half> v79;
      uint32_t v80 = v74 + v34;
      int32_t v81 = static_cast<int32_t>(v80);
      __gm__ half* v82 = reinterpret_cast<__gm__ half*>(v3);
      v79.SetGlobalBuffer(v82 + v81);
      AscendC::DataCopy(v78, v79, v76);
      v21.EnQue(v78);
      AscendC::LocalTensor<half> v83 = v21.DeQue<half>();
      uint32_t v84 = v76 * v64;
      uint32_t v85 = v84 * c2_idx;
      v18.InitBuffer(v69, v85);
      v18.InitBuffer(v22, c1_i32, v85);
      v18.InitBuffer(v68, v85);
      AscendC::LocalTensor<half> v86 = v68.Get<half>();
      int32_t v87 = static_cast<int32_t>(v76);
      int32_t v88 = static_cast<int32_t>(v64);
      v18.InitBuffer(v67, v85);
      AscendC::LocalTensor<half> v89 = v67.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v87, (uint32_t)v88};
        uint32_t _afir_ss[2] = {(uint32_t)v87, (uint32_t)c1_i32};
        AscendC::Broadcast<half, 2, 1>(v89, v83, _afir_ds, _afir_ss);
      };
      AscendC::GlobalTensor<half> v90;
      uint32_t v91 = v80 * v64;
      int32_t v92 = static_cast<int32_t>(v91);
      __gm__ half* v93 = reinterpret_cast<__gm__ half*>(v4);
      v90.SetGlobalBuffer(v93 + v92);
      v18.InitBuffer(v66, v85);
      v18.InitBuffer(v65, c1_i32, v85);
      AscendC::LocalTensor<half> v94 = v65.AllocTensor<half>();
      AscendC::DataCopy(v94, v90, v84);
      v65.EnQue(v94);
      AscendC::LocalTensor<half> v95 = v65.DeQue<half>();
      AscendC::Mul(v86, v89, v95, v84);
      v22.EnQue(v86);
      AscendC::LocalTensor<half> v96 = v22.DeQue<half>();
      AscendC::GlobalTensor<half> v97;
      uint32_t v98 = v80 + v25;
      uint32_t v99 = v98 * v26;
      int32_t v100 = static_cast<int32_t>(v99);
      __gm__ half* v101 = reinterpret_cast<__gm__ half*>(v5);
      v97.SetGlobalBuffer(v101 + v100);
      AscendC::DataCopy(v97, v96, v84);
      v22.FreeTensor(v96);
      v21.FreeTensor(v83);
    }
  }
  return;
}
