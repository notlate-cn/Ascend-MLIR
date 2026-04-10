#include "kernel_operator.h"
#include "adv_api/broadcast/broadcast.h"
#include "adv_api/reduce/reduce.h"

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t Tb_M;
  int64_t Tb_N;
  int64_t t_K;
  int64_t dim_arg3_0;
  int64_t dim_arg3_1;
  int64_t dim_arg0_1;
  int64_t dim_arg0_0;
  int64_t dim_arg1_0;
  int64_t dim_arg1_1;
  int64_t dim_arg2_0;
  int64_t dim_arg2_1;
};

extern "C" __global__ __aicore__ void fc_relu(
  GM_ADDR v1,
  GM_ADDR v2,
  GM_ADDR v3,
  GM_ADDR v4,
  GM_ADDR v5,
  TilingData v6
) {
  constexpr float c0_f32 = (float)0.0e+00;
  constexpr uint32_t c4_idx = 4;
  constexpr int32_t c1_i32 = 1;
  constexpr uint32_t c16_idx = 16;
  constexpr int16_t c0_i16 = 0;
  constexpr bool c0_i1 = false;
  constexpr int8_t c0_i8 = 0;
  constexpr int16_t c1_i16 = 1;
  constexpr bool cm1_i1 = true;
  constexpr uint32_t c0_idx = 0;
  int64_t v7 = v6.TB_M;
  int64_t v8 = v6.TB_N;
  int64_t v9 = v6.Tb_M;
  int64_t v10 = v6.Tb_N;
  int64_t v11 = v6.t_K;
  int64_t v12 = v6.dim_arg3_0;
  int64_t v13 = v6.dim_arg3_1;
  int64_t v14 = v6.dim_arg0_1;
  int64_t v15 = v6.dim_arg0_0;
  int64_t v16 = v6.dim_arg1_0;
  int64_t v17 = v6.dim_arg1_1;
  int64_t v18 = v6.dim_arg2_0;
  int64_t v19 = v6.dim_arg2_1;
  AscendC::TPipe v20;
  AscendC::TQue<AscendC::TPosition::A1, 1> v21;
  AscendC::TQue<AscendC::TPosition::B1, 1> v22;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v23;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v24;
  AscendC::TQue<AscendC::TPosition::CO1, 1> v25;
  AscendC::TQue<AscendC::TPosition::A2, 1> v26;
  AscendC::TQue<AscendC::TPosition::B2, 1> v27;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v28;
  AscendC::TQue<AscendC::TPosition::VECCALC, 1> v29;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v30;
  uint32_t v31 = static_cast<uint32_t>(v11);
  uint32_t v32 = static_cast<uint32_t>(v10);
  uint32_t v33 = static_cast<uint32_t>(v9);
  uint32_t v34 = static_cast<uint32_t>(v8);
  uint32_t v35 = static_cast<uint32_t>(v7);
  uint32_t v36 = static_cast<uint32_t>(v12);
  uint32_t v37 = static_cast<uint32_t>(v13);
  AscendC::TBuf<AscendC::TPosition::VECIN> v38;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v39;
  AscendC::TBuf<AscendC::TPosition::VECIN> v40;
  AscendC::TBuf<AscendC::TPosition::B2> v41;
  AscendC::TBuf<AscendC::TPosition::A2> v42;
  AscendC::TBuf<AscendC::TPosition::CO1> v43;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v44;
  AscendC::TBuf<AscendC::TPosition::VECIN> v45;
  AscendC::TBuf<AscendC::TPosition::B1> v46;
  AscendC::TBuf<AscendC::TPosition::A1> v47;
  uint32_t v48 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v49 = v48 * v35;
  bool v50 = v49 < v36;
  if (v50) {
    for (uint32_t v51 = c0_idx; v51 < v37; v51 += v34) {
      uint32_t v52 = v36 - v49;
      uint32_t v53 = ((v35 < v52) ? (v35) : (v52));
      uint32_t v54 = v37 - v51;
      uint32_t v55 = ((v54 < v34) ? (v54) : (v34));
      uint32_t v56 = static_cast<uint32_t>(v14);
      uint32_t v57 = v53 * v56;
      uint32_t v58 = v57 * c4_idx;
      v20.InitBuffer(v47, v58);
      v20.InitBuffer(v21, c1_i32, v58);
      AscendC::LocalTensor<float> v59 = v21.AllocTensor<float>();
      AscendC::GlobalTensor<float> v60;
      uint32_t v61 = v49 * v56;
      int32_t v62 = static_cast<int32_t>(v61);
      __gm__ float* v63 = reinterpret_cast<__gm__ float*>(v1);
      v60.SetGlobalBuffer(v63 + v62);
      uint32_t v64 = v56 / c16_idx;
      int16_t v65 = static_cast<int16_t>(v53);
      int16_t v66 = static_cast<int16_t>(v64);
      int16_t v67 = static_cast<int16_t>(v56);
      AscendC::Nd2NzParams v68{static_cast<uint16_t>(v65), static_cast<uint16_t>(v66), static_cast<uint16_t>(v65), static_cast<uint16_t>(v67), static_cast<uint16_t>(v65), static_cast<uint16_t>(c0_i16), static_cast<uint16_t>(v65), static_cast<uint16_t>(c0_i16)};
      AscendC::DataCopy(v59, v60, v68);
      v21.EnQue(v59);
      uint32_t v69 = v56 * v55;
      uint32_t v70 = v69 * c4_idx;
      v20.InitBuffer(v46, v70);
      v20.InitBuffer(v22, c1_i32, v70);
      AscendC::LocalTensor<float> v71 = v22.AllocTensor<float>();
      AscendC::GlobalTensor<float> v72;
      int32_t v73 = static_cast<int32_t>(v51);
      __gm__ float* v74 = reinterpret_cast<__gm__ float*>(v2);
      v72.SetGlobalBuffer(v74 + v73);
      uint32_t v75 = v55 / c16_idx;
      int16_t v76 = static_cast<int16_t>(v75);
      int16_t v77 = static_cast<int16_t>(v55);
      AscendC::Nd2NzParams v78{static_cast<uint16_t>(v67), static_cast<uint16_t>(v76), static_cast<uint16_t>(v67), static_cast<uint16_t>(v77), static_cast<uint16_t>(v67), static_cast<uint16_t>(c0_i16), static_cast<uint16_t>(v67), static_cast<uint16_t>(c0_i16)};
      AscendC::DataCopy(v71, v72, v78);
      v22.EnQue(v71);
      uint32_t v79 = v53 * v55;
      uint32_t v80 = v79 * c4_idx;
      v20.InitBuffer(v45, v80);
      v20.InitBuffer(v23, c1_i32, v80);
      AscendC::LocalTensor<float> v81 = v23.AllocTensor<float>();
      AscendC::GlobalTensor<float> v82;
      uint32_t v83 = static_cast<uint32_t>(v19);
      uint32_t v84 = v49 * v83;
      uint32_t v85 = v84 + v51;
      int32_t v86 = static_cast<int32_t>(v85);
      __gm__ float* v87 = reinterpret_cast<__gm__ float*>(v3);
      v82.SetGlobalBuffer(v87 + v86);
      AscendC::DataCopy(v81, v82, v79);
      v23.EnQue(v81);
      AscendC::LocalTensor<float> v88 = v23.DeQue<float>();
      v20.InitBuffer(v44, v80);
      v20.InitBuffer(v24, c1_i32, v80);
      AscendC::LocalTensor<float> v89 = v21.DeQue<float>();
      AscendC::LocalTensor<float> v90 = v22.DeQue<float>();
      for (uint32_t v91 = c0_idx; v91 < v53; v91 += v33) {
        AscendC::LocalTensor<float> v92 = v24.AllocTensor<float>();
        for (uint32_t v93 = c0_idx; v93 < v55; v93 += v32) {
          uint32_t v94 = v53 - v91;
          uint32_t v95 = ((v94 < v33) ? (v94) : (v33));
          uint32_t v96 = v55 - v93;
          uint32_t v97 = ((v96 < v32) ? (v96) : (v32));
          uint32_t v98 = v95 * v97;
          uint32_t v99 = v98 * c4_idx;
          v20.InitBuffer(v43, v99);
          v20.InitBuffer(v25, c1_i32, v99);
          AscendC::LocalTensor<float> v100 = v25.AllocTensor<float>();
          for (uint32_t v101 = c0_idx; v101 < v56; v101 += v31) {
            uint32_t v102 = v56 - v101;
            uint32_t v103 = ((v102 < v31) ? (v102) : (v31));
            uint32_t v104 = v95 * v103;
            uint32_t v105 = v104 * c4_idx;
            v20.InitBuffer(v42, v105);
            v20.InitBuffer(v26, c1_i32, v105);
            AscendC::LocalTensor<float> v106 = v26.AllocTensor<float>();
            uint32_t v107 = v95 / c16_idx;
            int16_t v108 = static_cast<int16_t>(v107);
            uint32_t v109 = v103 / c16_idx;
            int64_t v110 = static_cast<int64_t>(v109);
            int8_t v111 = static_cast<int8_t>(v110);
            AscendC::LoadData2DParams v112{static_cast<uint16_t>(c0_i16), static_cast<uint8_t>(v111), static_cast<uint16_t>(v108), static_cast<uint8_t>(c0_i16), static_cast<uint16_t>(c0_i16), c0_i1, static_cast<uint8_t>(c0_i8)};
            AscendC::LoadData(v106, v89, v112);
            v26.EnQue(v106);
            uint32_t v113 = v103 * v97;
            uint32_t v114 = v113 * c4_idx;
            v20.InitBuffer(v41, v114);
            v20.InitBuffer(v27, c1_i32, v114);
            AscendC::LocalTensor<float> v115 = v27.AllocTensor<float>();
            AscendC::LoadData2dTransposeParams v116{static_cast<uint16_t>(c0_i16), static_cast<uint8_t>(v111), static_cast<uint16_t>(c1_i16), static_cast<uint16_t>(c0_i16), static_cast<uint16_t>(c0_i16), static_cast<uint8_t>(c0_i8)};
            AscendC::LoadDataWithTranspose(v115, v90, v116);
            v27.EnQue(v115);
            AscendC::LocalTensor<float> v117 = v26.DeQue<float>();
            AscendC::LocalTensor<float> v118 = v27.DeQue<float>();
            int16_t v119 = static_cast<int16_t>(v95);
            int16_t v120 = static_cast<int16_t>(v103);
            int16_t v121 = static_cast<int16_t>(v97);
            AscendC::MmadParams v122{static_cast<uint16_t>(v119), static_cast<uint16_t>(v121), static_cast<uint16_t>(v120), static_cast<uint8_t>(c0_i8), static_cast<bool>(c0_i8), static_cast<bool>(c0_i8)};
            AscendC::Mmad(v100, v117, v118, v122);
            v26.FreeTensor(v117);
            v27.FreeTensor(v118);
          }
          v25.EnQue(v100);
          v20.InitBuffer(v40, v99);
          v20.InitBuffer(v28, c1_i32, v99);
          AscendC::LocalTensor<float> v123 = v25.DeQue<float>();
          AscendC::LocalTensor<float> v124 = v28.AllocTensor<float>();
          AscendC::DataCopyCO12DstParams v125;
          AscendC::DataCopy(v124, v123, v125);
          v28.EnQue(v124);
          v25.FreeTensor(v123);
          v20.InitBuffer(v39, v99);
          v20.InitBuffer(v29, c1_i32, v99);
          AscendC::LocalTensor<float> v126 = v28.DeQue<float>();
          uint32_t v127 = v91 * v55;
          uint32_t v128 = v127 + v93;
          uint32_t v129 = v128 * c4_idx;
          AscendC::LocalTensor<float> v130 = v45.GetWithOffset<float>(v99, v129);
          AscendC::LocalTensor<float> v131 = v29.AllocTensor<float>();
          AscendC::Add(v131, v126, v130, v98);
          v29.EnQue(v131);
          v28.FreeTensor(v126);
          v20.InitBuffer(v38, v99);
          v20.InitBuffer(v30, c1_i32, v99);
          AscendC::LocalTensor<float> v132 = v30.AllocTensor<float>();
          AscendC::Duplicate(v132, c0_f32, v98);
          v30.EnQue(v132);
          AscendC::LocalTensor<float> v133 = v29.DeQue<float>();
          AscendC::LocalTensor<float> v134 = v30.DeQue<float>();
          AscendC::LocalTensor<float> v135 = v44.GetWithOffset<float>(v99, v129);
          AscendC::Max(v135, v133, v134, v98);
          v29.FreeTensor(v133);
          v30.FreeTensor(v134);
        }
        v24.EnQue(v92);
      }
      v22.FreeTensor(v90);
      v21.FreeTensor(v89);
      AscendC::LocalTensor<float> v136 = v24.DeQue<float>();
      AscendC::GlobalTensor<float> v137;
      uint32_t v138 = v49 * v37;
      uint32_t v139 = v138 + v51;
      int32_t v140 = static_cast<int32_t>(v139);
      __gm__ float* v141 = reinterpret_cast<__gm__ float*>(v4);
      v137.SetGlobalBuffer(v141 + v140);
      AscendC::DataCopy(v137, v136, v79);
      v24.FreeTensor(v136);
      v23.FreeTensor(v88);
    }
  }
  return;
}
