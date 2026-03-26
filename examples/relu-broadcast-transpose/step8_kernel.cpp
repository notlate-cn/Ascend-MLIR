extern "C"  __global__ __aicore__ void relu_transpose_broadcast_add(half* v1, half* v2, __gm__ TilingData* v3, half* v4) {
  constexpr int32_t c0_i32 = 0;
  half c0_f16 = 0.0e+00;
  constexpr uint32_t c0_idx = 0;
  constexpr uint32_t c2_idx = 2;
  constexpr int32_t c1_i32 = 1;
  TilingData v5;
  for (size_t i = 0; i < sizeof(v5); i++) {
    auto byte = reinterpret_cast<__gm__ uint8_t*>(v3)[i];
    reinterpret_cast<uint8_t*>(&v5)[i] = byte;
  };
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
  AscendC::TBuf<AscendC::TPosition::VECOUT> v23;
  AscendC::TBuf<AscendC::TPosition::VECIN> v24;
  uint32_t v25 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v26 = v25 * v16;
  bool v27 = v26 < v18;
  if (v27) {
    uint32_t v28 = v18 - v26;
    uint32_t v29 = ((v16 < v28) ? (v16) : (v28));
    for (uint32_t v30 = c0_idx; v30 < v29; v30 += v15) {
      uint32_t v31 = v29 - v30;
      uint32_t v32 = ((v31 < v15) ? (v31) : (v15));
      uint32_t v33 = v17 * c2_idx;
      v12.InitBuffer(v24, v33);
      v12.InitBuffer(v13, c1_i32, v33);
      AscendC::LocalTensor<half> v34 = v13.AllocTensor<half>();
      AscendC::GlobalTensor<half> v35;
      __gm__ half* v36 = reinterpret_cast<__gm__ half*>(v1);
      v35.SetGlobalBuffer(v36, c0_i32);
      AscendC::DataCopy(v34, v35, v17);
      v13.EnQue(v34);
      AscendC::LocalTensor<half> v37 = v13.DeQue<half>();
      uint32_t v38 = v32 * v17;
      uint32_t v39 = v38 * c2_idx;
      v12.InitBuffer(v23, v39);
      v12.InitBuffer(v14, c1_i32, v39);
      v12.InitBuffer(v22, v33);
      AscendC::LocalTensor<half> v40 = v22.Get<half>();
      AscendC::GlobalTensor<half> v41;
      uint32_t v42 = v30 + v26;
      uint32_t v43 = static_cast<uint32_t>(v11);
      uint32_t v44 = v42 * v43;
      int32_t v45 = static_cast<int32_t>(v44);
      __gm__ half* v46 = reinterpret_cast<__gm__ half*>(v2);
      v41.SetGlobalBuffer(v46, v45);
      v12.InitBuffer(v21, v33);
      v12.InitBuffer(v20, c1_i32, v33);
      AscendC::LocalTensor<half> v47 = v20.AllocTensor<half>();
      AscendC::DataCopy(v47, v41, v17);
      v20.EnQue(v47);
      AscendC::LocalTensor<half> v48 = v20.DeQue<half>();
      v12.InitBuffer(v19, v33);
      AscendC::LocalTensor<half> v49 = v19.Get<half>();
      AscendC::Duplicate(v49, c0_f16, v17);
      AscendC::Max(v40, v37, v49, v17);
      AscendC::Add(v40, v40, v48, v17);
      v14.EnQue(v40);
      AscendC::LocalTensor<half> v50 = v14.DeQue<half>();
      AscendC::GlobalTensor<half> v51;
      uint32_t v52 = v42 * v17;
      int32_t v53 = static_cast<int32_t>(v52);
      __gm__ half* v54 = reinterpret_cast<__gm__ half*>(v4);
      v51.SetGlobalBuffer(v54, v53);
      AscendC::DataCopy(v51, v50, v38);
      v14.FreeTensor(v50);
      v13.FreeTensor(v37);
    }
  }
  return;
}

