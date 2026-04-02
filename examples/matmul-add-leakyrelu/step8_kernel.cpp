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

extern "C" __global__ __aicore__ void matmul_add_leakyrelu(
  GM_ADDR v1,
  GM_ADDR v2,
  GM_ADDR v3,
  GM_ADDR v4,
  GM_ADDR v5,
  GM_ADDR v6,
  TilingData v7
) {
  constexpr float c0_00100000005_f32 = (float)1.000000050e-03;
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c1_i32 = 1;
  constexpr uint32_t c16_idx = 16;
  constexpr int16_t c0_i16 = 0;
  constexpr uint32_t c4_idx = 4;
  constexpr bool c0_i1 = false;
  constexpr int8_t c0_i8 = 0;
  constexpr int16_t c1_i16 = 1;
  constexpr bool cm1_i1 = true;
  constexpr uint32_t c0_idx = 0;
  int64_t v8 = v7.TB_M;
  int64_t v9 = v7.TB_N;
  int64_t v10 = v7.Tb_M;
  int64_t v11 = v7.Tb_N;
  int64_t v12 = v7.t_K;
  int64_t v13 = v7.dim_arg3_0;
  int64_t v14 = v7.dim_arg3_1;
  int64_t v15 = v7.dim_arg0_1;
  int64_t v16 = v7.dim_arg0_0;
  int64_t v17 = v7.dim_arg1_0;
  int64_t v18 = v7.dim_arg1_1;
  int64_t v19 = v7.dim_arg2_0;
  int64_t v20 = v7.dim_arg2_1;
  AscendC::TPipe v21;
  AscendC::TQue<AscendC::TPosition::A1, 1> v22;
  AscendC::TQue<AscendC::TPosition::B1, 1> v23;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v24;
  AscendC::TQue<AscendC::TPosition::VECCALC, 1> v25;
  AscendC::TQue<AscendC::TPosition::CO1, 1> v26;
  AscendC::TQue<AscendC::TPosition::A2, 1> v27;
  AscendC::TQue<AscendC::TPosition::B2, 1> v28;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v29;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v30;
  uint32_t v31 = static_cast<uint32_t>(v12);
  uint32_t v32 = static_cast<uint32_t>(v11);
  uint32_t v33 = static_cast<uint32_t>(v10);
  uint32_t v34 = static_cast<uint32_t>(v9);
  uint32_t v35 = static_cast<uint32_t>(v8);
  uint32_t v36 = static_cast<uint32_t>(v13);
  uint32_t v37 = static_cast<uint32_t>(v14);
  AscendC::TBuf<AscendC::TPosition::VECCALC> v38;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v39;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v40;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v41;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v42;
  AscendC::TBuf<AscendC::TPosition::VECIN> v43;
  AscendC::TBuf<AscendC::TPosition::CO1> v44;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v45;
  AscendC::TBuf<AscendC::TPosition::VECIN> v46;
  AscendC::TBuf<AscendC::TPosition::B2> v47;
  AscendC::TBuf<AscendC::TPosition::A2> v48;
  AscendC::TBuf<AscendC::TPosition::CO1> v49;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v50;
  AscendC::TBuf<AscendC::TPosition::VECIN> v51;
  AscendC::TBuf<AscendC::TPosition::B1> v52;
  AscendC::TBuf<AscendC::TPosition::A1> v53;
  uint32_t v54 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v55 = v54 * v35;
  bool v56 = v55 < v36;
  if (v56) {
    for (uint32_t v57 = c0_idx; v57 < v37; v57 += v34) {
      uint32_t v58 = v36 - v55;
      uint32_t v59 = ((v35 < v58) ? (v35) : (v58));
      uint32_t v60 = v37 - v57;
      uint32_t v61 = ((v60 < v34) ? (v60) : (v34));
      uint32_t v62 = static_cast<uint32_t>(v15);
      uint32_t v63 = v59 * v62;
      uint32_t v64 = v63 * c2_idx;
      v21.InitBuffer(v53, v64);
      v21.InitBuffer(v22, c1_i32, v64);
      AscendC::LocalTensor<half> v65 = v22.AllocTensor<half>();
      AscendC::GlobalTensor<half> v66;
      uint32_t v67 = v55 * v62;
      int32_t v68 = static_cast<int32_t>(v67);
      __gm__ half* v69 = reinterpret_cast<__gm__ half*>(v1);
      v66.SetGlobalBuffer(v69 + v68);
      uint32_t v70 = v62 / c16_idx;
      int16_t v71 = static_cast<int16_t>(v59);
      int16_t v72 = static_cast<int16_t>(v70);
      int16_t v73 = static_cast<int16_t>(v62);
      AscendC::Nd2NzParams v74{v71, v72, v71, v73, v71, c0_i16, v71, c0_i16};
      AscendC::DataCopy(v65, v66, v74);
      v22.EnQue(v65);
      uint32_t v75 = v62 * v61;
      uint32_t v76 = v75 * c2_idx;
      v21.InitBuffer(v52, v76);
      v21.InitBuffer(v23, c1_i32, v76);
      AscendC::LocalTensor<half> v77 = v23.AllocTensor<half>();
      AscendC::GlobalTensor<half> v78;
      int32_t v79 = static_cast<int32_t>(v57);
      __gm__ half* v80 = reinterpret_cast<__gm__ half*>(v2);
      v78.SetGlobalBuffer(v80 + v79);
      uint32_t v81 = v61 / c16_idx;
      int16_t v82 = static_cast<int16_t>(v81);
      int16_t v83 = static_cast<int16_t>(v61);
      AscendC::Nd2NzParams v84{v73, v82, v73, v83, v73, c0_i16, v73, c0_i16};
      AscendC::DataCopy(v77, v78, v84);
      v23.EnQue(v77);
      uint32_t v85 = v61 * c4_idx;
      v21.InitBuffer(v51, v85);
      v21.InitBuffer(v24, c1_i32, v85);
      AscendC::LocalTensor<float> v86 = v24.AllocTensor<float>();
      AscendC::GlobalTensor<float> v87;
      __gm__ float* v88 = reinterpret_cast<__gm__ float*>(v3);
      v87.SetGlobalBuffer(v88 + v79);
      AscendC::DataCopy(v86, v87, v61);
      v24.EnQue(v86);
      AscendC::LocalTensor<float> v89 = v24.DeQue<float>();
      uint32_t v90 = v59 * v61;
      uint32_t v91 = v90 * c4_idx;
      v21.InitBuffer(v50, v91);
      v21.InitBuffer(v25, c1_i32, v91);
      AscendC::LocalTensor<half> v92 = v22.DeQue<half>();
      AscendC::LocalTensor<half> v93 = v23.DeQue<half>();
      for (uint32_t v94 = c0_idx; v94 < v59; v94 += v33) {
        for (uint32_t v95 = c0_idx; v95 < v61; v95 += v32) {
          uint32_t v96 = v59 - v94;
          uint32_t v97 = ((v96 < v33) ? (v96) : (v33));
          uint32_t v98 = v61 - v95;
          uint32_t v99 = ((v98 < v32) ? (v98) : (v32));
          uint32_t v100 = v97 * v99;
          uint32_t v101 = v100 * c4_idx;
          v21.InitBuffer(v49, v101);
          v21.InitBuffer(v26, c1_i32, v101);
          AscendC::LocalTensor<half> v102 = v26.AllocTensor<half>();
          for (uint32_t v103 = c0_idx; v103 < v62; v103 += v31) {
            uint32_t v104 = v62 - v103;
            uint32_t v105 = ((v104 < v31) ? (v104) : (v31));
            uint32_t v106 = v97 * v105;
            uint32_t v107 = v106 * c2_idx;
            v21.InitBuffer(v48, v107);
            v21.InitBuffer(v27, c1_i32, v107);
            AscendC::LocalTensor<half> v108 = v27.AllocTensor<half>();
            uint32_t v109 = v97 / c16_idx;
            int16_t v110 = static_cast<int16_t>(v109);
            uint32_t v111 = v105 / c16_idx;
            int64_t v112 = static_cast<int64_t>(v111);
            int8_t v113 = static_cast<int8_t>(v112);
            AscendC::LoadData2DParams v114{c0_i16, v113, v110, c0_i16, c0_i16, c0_i1, c0_i8};
            AscendC::LoadData(v108, v92, v114);
            v27.EnQue(v108);
            uint32_t v115 = v105 * v99;
            uint32_t v116 = v115 * c2_idx;
            v21.InitBuffer(v47, v116);
            v21.InitBuffer(v28, c1_i32, v116);
            AscendC::LocalTensor<half> v117 = v28.AllocTensor<half>();
            AscendC::LoadData2dTransposeParams v118{c0_i16, v113, c1_i16, c0_i16, c0_i16, cm1_i1, c0_i8};
            AscendC::LoadDataWithTranspose(v117, v93, v118);
            v28.EnQue(v117);
            AscendC::LocalTensor<half> v119 = v27.DeQue<half>();
            AscendC::LocalTensor<half> v120 = v28.DeQue<half>();
            int16_t v121 = static_cast<int16_t>(v97);
            int16_t v122 = static_cast<int16_t>(v105);
            int16_t v123 = static_cast<int16_t>(v99);
            AscendC::MmadParams v124{v121, v123, v122, c0_i8, c0_i8, c0_i8};
            AscendC::Mmad(v102, v119, v120, v124);
            v27.FreeTensor(v119);
            v28.FreeTensor(v120);
          }
          v26.EnQue(v102);
          v21.InitBuffer(v46, v101);
          v21.InitBuffer(v29, c1_i32, v101);
          AscendC::LocalTensor<float> v125 = v26.DeQue<float>();
          AscendC::LocalTensor<float> v126 = v29.AllocTensor<float>();
          AscendC::DataCopyCO12DstParams v127;
          AscendC::DataCopy(v126, v125, v127);
          v29.EnQue(v126);
          v26.FreeTensor(v125);
          v21.InitBuffer(v45, v101);
          AscendC::LocalTensor<float> v128 = v45.Get<float>();
          AscendC::LocalTensor<float> v129 = v44.Get<float>();
          AscendC::GlobalTensor<float> v130;
          uint32_t v131 = v95 + v57;
          int32_t v132 = static_cast<int32_t>(v131);
          v130.SetGlobalBuffer(v88 + v132);
          uint32_t v133 = v99 * c4_idx;
          v21.InitBuffer(v43, v133);
          v21.InitBuffer(v42, c1_i32, v133);
          AscendC::LocalTensor<float> v134 = v42.AllocTensor<float>();
          AscendC::DataCopy(v134, v130, v99);
          v42.EnQue(v134);
          AscendC::LocalTensor<float> v135 = v42.DeQue<float>();
          int32_t v136 = static_cast<int32_t>(v97);
          int32_t v137 = static_cast<int32_t>(v99);
          v21.InitBuffer(v41, v101);
          AscendC::LocalTensor<float> v138 = v41.Get<float>();
          {
            uint32_t _afir_ds[2] = {(uint32_t)v136, (uint32_t)v137};
            uint32_t _afir_ss[2] = {(uint32_t)c1_i32, (uint32_t)v137};
            AscendC::Broadcast<half, 2, 0>(v138, v135, _afir_ds, _afir_ss);
          };
          AscendC::Add(v128, v129, v138, v100);
          v25.EnQue(v128);
          v21.InitBuffer(v40, v101);
          v21.InitBuffer(v30, c1_i32, v101);
          v21.InitBuffer(v39, v101);
          AscendC::LocalTensor<float> v139 = v39.Get<float>();
          AscendC::LocalTensor<float> v140 = v25.DeQue<float>();
          v21.InitBuffer(v38, v101);
          AscendC::LocalTensor<float> v141 = v38.Get<float>();
          AscendC::Duplicate(v141, c0_00100000005_f32, v100);
          AscendC::Mul(v139, v140, v141, v100);
          AscendC::Max(v139, v140, v139, v100);
          v30.EnQue(v139);
          AscendC::LocalTensor<float> v142 = v30.DeQue<float>();
          AscendC::GlobalTensor<float> v143;
          uint32_t v144 = v94 + v55;
          uint32_t v145 = v144 * v37;
          uint32_t v146 = v145 + v131;
          int32_t v147 = static_cast<int32_t>(v146);
          __gm__ float* v148 = reinterpret_cast<__gm__ float*>(v5);
          v143.SetGlobalBuffer(v148 + v147);
          AscendC::DataCopy(v143, v142, v100);
          v30.FreeTensor(v142);
        }
      }
      v23.FreeTensor(v93);
      v22.FreeTensor(v92);
      v24.FreeTensor(v89);
    }
  }
  return;
}
