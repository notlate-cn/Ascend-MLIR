extern "C"  __global__ __aicore__ void relu_index_select_add(half* v1, int64_t* v2, half* v3, __gm__ TilingData* v4, half* v5) {
  half c0_f16 = 0.0e+00;
  constexpr uint32_t c0_idx = 0;
  constexpr uint32_t c8_idx = 8;
  constexpr int32_t c1_i32 = 1;
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c0_i32 = 0;
  constexpr uint32_t c1_idx = 1;
  TilingData v6;
  for (size_t i = 0; i < sizeof(v6); i++) {
    auto byte = reinterpret_cast<__gm__ uint8_t*>(v4)[i];
    reinterpret_cast<uint8_t*>(&v6)[i] = byte;
  };
  int64_t v7 = v6.TB_M;
  int64_t v8 = v6.TB_N;
  int64_t v9 = v6.dim_arg0_0;
  int64_t v10 = v6.dim_arg1_0;
  int64_t v11 = v6.dim_arg0_1;
  int64_t v12 = v6.dim_arg1_1;
  AscendC::TPipe v13;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v14;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v15;
  uint32_t v16 = static_cast<uint32_t>(v8);
  uint32_t v17 = static_cast<uint32_t>(v7);
  uint32_t v18 = static_cast<uint32_t>(v9);
  uint32_t v19 = static_cast<uint32_t>(v10);
  AscendC::TBuf<AscendC::TPosition::GM> v20;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v21;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v22;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v23;
  AscendC::TBuf<AscendC::TPosition::VECIN> v24;
  uint32_t v25 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v26 = v25 * v17;
  bool v27 = v26 < v18;
  if (v27) {
    uint32_t v28 = v18 - v26;
    uint32_t v29 = ((v17 < v28) ? (v17) : (v28));
    for (uint32_t v30 = c0_idx; v30 < v29; v30 += v16) {
      uint32_t v31 = v29 - v30;
      uint32_t v32 = ((v31 < v16) ? (v31) : (v16));
      uint32_t v33 = v19 * c8_idx;
      v13.InitBuffer(v24, v33);
      v13.InitBuffer(v14, c1_i32, v33);
      AscendC::LocalTensor<int64_t> v34 = v14.AllocTensor<int64_t>();
      AscendC::GlobalTensor<int64_t> v35;
      __gm__ int64_t* v36 = reinterpret_cast<__gm__ int64_t*>(v2);
      v35.SetGlobalBuffer(v36, c0_i32);
      AscendC::DataCopy(v34, v35, v19);
      v14.EnQue(v34);
      AscendC::LocalTensor<int64_t> v37 = v14.DeQue<int64_t>();
      uint32_t v38 = v32 * v19;
      uint32_t v39 = v38 * c2_idx;
      v13.InitBuffer(v23, v39);
      v13.InitBuffer(v15, c1_i32, v39);
      uint32_t v40 = static_cast<uint32_t>(v11);
      int32_t v41 = static_cast<int32_t>(v19);
      AscendC::LocalTensor<half> v42 = v15.AllocTensor<half>();
      uint32_t v43 = v19 * c2_idx;
      AscendC::GlobalTensor<half> v44;
      v44.SetGlobalBuffer(v1);
      uint32_t v45 = v40 * c2_idx;
      v13.InitBuffer(v22, v45);
      AscendC::LocalTensor<half> v46 = v22.Get<half>();
      for (uint32_t v47 = c0_idx; v47 < v32; v47 += c1_idx) {
        uint32_t v48 = v47 + v30;
        uint32_t v49 = v48 + v26;
        uint32_t v50 = v49 * v40;
        AscendC::GlobalTensor<half> v51 = v44(v50);
        AscendC::LocalTensor<half> v52 = v22.Get<half>();
        AscendC::DataCopy(v52, v51, v40);
        uint32_t v53 = v47 * v43;
        AscendC::LocalTensor<half> v54 = v23.GetWithOffset<half>(v43, v53);
        AscendC::Gather(v54, v52, v37, c0_i32, v41);
        v13.InitBuffer(v21, v43);
        AscendC::LocalTensor<half> v55 = v21.Get<half>();
        AscendC::Duplicate(v55, c0_f16, v41);
        AscendC::Max(v54, v54, v55, v41);
        AscendC::LocalTensor<half> v56 = v20.Get<half>();
        AscendC::Add(v54, v54, v56, v41);
      }
      v15.EnQue(v42);
      AscendC::LocalTensor<half> v57 = v15.DeQue<half>();
      AscendC::GlobalTensor<half> v58;
      uint32_t v59 = v30 + v26;
      uint32_t v60 = v59 * v19;
      int32_t v61 = static_cast<int32_t>(v60);
      __gm__ half* v62 = reinterpret_cast<__gm__ half*>(v5);
      v58.SetGlobalBuffer(v62, v61);
      AscendC::DataCopy(v58, v57, v38);
      v15.FreeTensor(v57);
      v14.FreeTensor(v37);
    }
  }
  return;
}

