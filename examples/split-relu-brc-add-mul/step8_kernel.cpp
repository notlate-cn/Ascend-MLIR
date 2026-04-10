#include "kernel_operator.h"
#include "adv_api/broadcast/broadcast.h"
#include "adv_api/reduce/reduce.h"

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_1;
  int64_t dim_arg1_0;
  int64_t dim_arg0_0;
  int64_t dim_arg1_1;
  int64_t dim_arg3_0;
  int64_t dim_arg3_1;
  int64_t dim_arg2_0;
  int64_t dim_arg2_1;
  int64_t dim_arg4_0;
  int64_t dim_arg4_1;
};

extern "C" __global__ __aicore__ void ewop_broadcast_split(
  GM_ADDR v1,
  GM_ADDR v2,
  GM_ADDR v3,
  GM_ADDR v4,
  GM_ADDR v5,
  GM_ADDR v6,
  GM_ADDR v7,
  TilingData v8
) {
  constexpr int32_t c0_i32 = 0;
  half c0_f16 = 0.0e+00;
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c1_i32 = 1;
  constexpr uint32_t c0_idx = 0;
  int64_t v9 = v8.TB_M;
  int64_t v10 = v8.TB_N;
  int64_t v11 = v8.dim_arg0_1;
  int64_t v12 = v8.dim_arg1_0;
  int64_t v13 = v8.dim_arg0_0;
  int64_t v14 = v8.dim_arg1_1;
  int64_t v15 = v8.dim_arg3_0;
  int64_t v16 = v8.dim_arg3_1;
  int64_t v17 = v8.dim_arg2_0;
  int64_t v18 = v8.dim_arg2_1;
  int64_t v19 = v8.dim_arg4_0;
  int64_t v20 = v8.dim_arg4_1;
  AscendC::TPipe v21;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v22;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v23;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v24;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v25;
  uint32_t v26 = static_cast<uint32_t>(v10);
  uint32_t v27 = static_cast<uint32_t>(v9);
  uint32_t v28 = static_cast<uint32_t>(v11);
  uint32_t v29 = static_cast<uint32_t>(v12);
  AscendC::TBuf<AscendC::TPosition::VECCALC> v30;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v31;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v32;
  AscendC::TBuf<AscendC::TPosition::VECIN> v33;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v34;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v35;
  AscendC::TBuf<AscendC::TPosition::VECIN> v36;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v37;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v38;
  AscendC::TBuf<AscendC::TPosition::VECIN> v39;
  uint32_t v40 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v41 = v40 * v27;
  bool v42 = v41 < v29;
  if (v42) {
    uint32_t v43 = v29 - v41;
    uint32_t v44 = ((v27 < v43) ? (v27) : (v43));
    for (uint32_t v45 = c0_idx; v45 < v44; v45 += v26) {
      uint32_t v46 = v44 - v45;
      uint32_t v47 = ((v46 < v26) ? (v46) : (v26));
      uint32_t v48 = v47 * v28;
      uint32_t v49 = v48 * c2_idx;
      v21.InitBuffer(v39, v49);
      v21.InitBuffer(v22, c1_i32, v49);
      AscendC::LocalTensor<half> v50 = v22.AllocTensor<half>();
      AscendC::GlobalTensor<half> v51;
      uint32_t v52 = v45 + v41;
      uint32_t v53 = v52 * v28;
      int32_t v54 = static_cast<int32_t>(v53);
      __gm__ half* v55 = reinterpret_cast<__gm__ half*>(v1);
      v51.SetGlobalBuffer(v55 + v54);
      AscendC::DataCopy(v50, v51, v48);
      v22.EnQue(v50);
      AscendC::LocalTensor<half> v56 = v22.DeQue<half>();
      v21.InitBuffer(v38, v49);
      v21.InitBuffer(v23, c1_i32, v49);
      v21.InitBuffer(v37, v49);
      AscendC::LocalTensor<half> v57 = v37.Get<half>();
      AscendC::GlobalTensor<half> v58;
      int32_t v59 = static_cast<int32_t>(v52);
      __gm__ half* v60 = reinterpret_cast<__gm__ half*>(v2);
      v58.SetGlobalBuffer(v60 + v59);
      uint32_t v61 = v47 * c2_idx;
      v21.InitBuffer(v36, v61);
      v21.InitBuffer(v35, c1_i32, v61);
      AscendC::LocalTensor<half> v62 = v35.AllocTensor<half>();
      AscendC::DataCopy(v62, v58, v47);
      v35.EnQue(v62);
      AscendC::LocalTensor<half> v63 = v35.DeQue<half>();
      int32_t v64 = static_cast<int32_t>(v47);
      int32_t v65 = static_cast<int32_t>(v28);
      v21.InitBuffer(v34, v49);
      AscendC::LocalTensor<half> v66 = v34.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v64, (uint32_t)v65};
        uint32_t _afir_ss[2] = {(uint32_t)v64, (uint32_t)c1_i32};
        AscendC::Broadcast<half, 2, 1>(v66, v63, _afir_ds, _afir_ss);
      };
      AscendC::GlobalTensor<half> v67;
      __gm__ half* v68 = reinterpret_cast<__gm__ half*>(v4);
      v67.SetGlobalBuffer(v68 + c0_i32);
      uint32_t v69 = v28 * c2_idx;
      v21.InitBuffer(v33, v69);
      v21.InitBuffer(v32, c1_i32, v69);
      AscendC::LocalTensor<half> v70 = v32.AllocTensor<half>();
      AscendC::DataCopy(v70, v67, v28);
      v32.EnQue(v70);
      AscendC::LocalTensor<half> v71 = v32.DeQue<half>();
      v21.InitBuffer(v31, v49);
      AscendC::LocalTensor<half> v72 = v31.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v64, (uint32_t)v65};
        uint32_t _afir_ss[2] = {(uint32_t)c1_i32, (uint32_t)v65};
        AscendC::Broadcast<half, 2, 0>(v72, v71, _afir_ds, _afir_ss);
      };
      v21.InitBuffer(v30, v49);
      AscendC::LocalTensor<half> v73 = v30.Get<half>();
      AscendC::Duplicate(v73, c0_f16, v48);
      AscendC::Max(v57, v56, v73, v48);
      AscendC::Add(v57, v57, v66, v48);
      AscendC::Mul(v57, v57, v72, v48);
      v23.EnQue(v57);
      AscendC::LocalTensor<half> v74 = v23.DeQue<half>();
      AscendC::GlobalTensor<half> v75;
      __gm__ half* v76 = reinterpret_cast<__gm__ half*>(v6);
      v75.SetGlobalBuffer(v76 + v54);
      AscendC::DataCopy(v75, v74, v48);
      v23.FreeTensor(v74);
      v22.FreeTensor(v56);
    }
  }
  AscendC::TBuf<AscendC::TPosition::VECCALC> v77;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v78;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v79;
  AscendC::TBuf<AscendC::TPosition::VECIN> v80;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v81;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v82;
  AscendC::TBuf<AscendC::TPosition::VECIN> v83;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v84;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v85;
  AscendC::TBuf<AscendC::TPosition::VECIN> v86;
  if (v42) {
    uint32_t v87 = v29 - v41;
    uint32_t v88 = ((v27 < v87) ? (v27) : (v87));
    for (uint32_t v89 = c0_idx; v89 < v88; v89 += v26) {
      uint32_t v90 = v88 - v89;
      uint32_t v91 = ((v90 < v26) ? (v90) : (v26));
      uint32_t v92 = v91 * v28;
      uint32_t v93 = v92 * c2_idx;
      v21.InitBuffer(v86, v93);
      v21.InitBuffer(v24, c1_i32, v93);
      AscendC::LocalTensor<half> v94 = v24.AllocTensor<half>();
      AscendC::GlobalTensor<half> v95;
      uint32_t v96 = v89 + v41;
      uint32_t v97 = v96 + v29;
      uint32_t v98 = v97 * v28;
      int32_t v99 = static_cast<int32_t>(v98);
      __gm__ half* v100 = reinterpret_cast<__gm__ half*>(v1);
      v95.SetGlobalBuffer(v100 + v99);
      AscendC::DataCopy(v94, v95, v92);
      v24.EnQue(v94);
      AscendC::LocalTensor<half> v101 = v24.DeQue<half>();
      v21.InitBuffer(v85, v93);
      v21.InitBuffer(v25, c1_i32, v93);
      v21.InitBuffer(v84, v93);
      AscendC::LocalTensor<half> v102 = v84.Get<half>();
      AscendC::GlobalTensor<half> v103;
      int32_t v104 = static_cast<int32_t>(v96);
      __gm__ half* v105 = reinterpret_cast<__gm__ half*>(v3);
      v103.SetGlobalBuffer(v105 + v104);
      uint32_t v106 = v91 * c2_idx;
      v21.InitBuffer(v83, v106);
      v21.InitBuffer(v82, c1_i32, v106);
      AscendC::LocalTensor<half> v107 = v82.AllocTensor<half>();
      AscendC::DataCopy(v107, v103, v91);
      v82.EnQue(v107);
      AscendC::LocalTensor<half> v108 = v82.DeQue<half>();
      int32_t v109 = static_cast<int32_t>(v91);
      int32_t v110 = static_cast<int32_t>(v28);
      v21.InitBuffer(v81, v93);
      AscendC::LocalTensor<half> v111 = v81.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v109, (uint32_t)v110};
        uint32_t _afir_ss[2] = {(uint32_t)v109, (uint32_t)c1_i32};
        AscendC::Broadcast<half, 2, 1>(v111, v108, _afir_ds, _afir_ss);
      };
      AscendC::GlobalTensor<half> v112;
      __gm__ half* v113 = reinterpret_cast<__gm__ half*>(v5);
      v112.SetGlobalBuffer(v113 + c0_i32);
      uint32_t v114 = v28 * c2_idx;
      v21.InitBuffer(v80, v114);
      v21.InitBuffer(v79, c1_i32, v114);
      AscendC::LocalTensor<half> v115 = v79.AllocTensor<half>();
      AscendC::DataCopy(v115, v112, v28);
      v79.EnQue(v115);
      AscendC::LocalTensor<half> v116 = v79.DeQue<half>();
      v21.InitBuffer(v78, v93);
      AscendC::LocalTensor<half> v117 = v78.Get<half>();
      {
        uint32_t _afir_ds[2] = {(uint32_t)v109, (uint32_t)v110};
        uint32_t _afir_ss[2] = {(uint32_t)c1_i32, (uint32_t)v110};
        AscendC::Broadcast<half, 2, 0>(v117, v116, _afir_ds, _afir_ss);
      };
      v21.InitBuffer(v77, v93);
      AscendC::LocalTensor<half> v118 = v77.Get<half>();
      AscendC::Duplicate(v118, c0_f16, v92);
      AscendC::Max(v102, v101, v118, v92);
      AscendC::Add(v102, v102, v111, v92);
      AscendC::Mul(v102, v102, v117, v92);
      v25.EnQue(v102);
      AscendC::LocalTensor<half> v119 = v25.DeQue<half>();
      AscendC::GlobalTensor<half> v120;
      __gm__ half* v121 = reinterpret_cast<__gm__ half*>(v6);
      v120.SetGlobalBuffer(v121 + v99);
      AscendC::DataCopy(v120, v119, v92);
      v25.FreeTensor(v119);
      v24.FreeTensor(v101);
    }
  }
  return;
}
