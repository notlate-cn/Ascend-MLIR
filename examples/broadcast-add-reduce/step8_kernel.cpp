extern "C"  __global__ __aicore__ void broadcast_add_reducesum(half* v1, half* v2, __gm__ TilingData* v3, half* v4) {
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
  int64_t v9 = v5.dim_arg1_1;
  AscendC::TPipe v10;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> v11;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> v12;
  uint32_t v13 = static_cast<uint32_t>(v7);
  uint32_t v14 = static_cast<uint32_t>(v6);
  uint32_t v15 = static_cast<uint32_t>(v8);
  uint32_t v16 = static_cast<uint32_t>(v9);
  AscendC::TBuf<AscendC::TPosition::VECCALC> v17;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v18;
  AscendC::TBuf<AscendC::TPosition::VECCALC> v19;
  AscendC::TBuf<AscendC::TPosition::VECOUT> v20;
  AscendC::TBuf<AscendC::TPosition::VECIN> v21;
  uint32_t v22 = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t v23 = v22 * v14;
  bool v24 = v23 < v15;
  if (v24) {
    uint32_t v25 = v15 - v23;
    uint32_t v26 = ((v14 < v25) ? (v14) : (v25));
    for (uint32_t v27 = c0_idx; v27 < v26; v27 += v13) {
      uint32_t v28 = v26 - v27;
      uint32_t v29 = ((v28 < v13) ? (v28) : (v13));
      uint32_t v30 = v29 * c2_idx;
      v10.InitBuffer(v21, v30);
      AscendC::LocalTensor<half> v31 = v11.AllocTensor<half>();
      AscendC::GlobalTensor<half> v32;
      uint32_t v33 = v27 + v23;
      int32_t v34 = static_cast<int32_t>(v33);
      __gm__ half* v35 = reinterpret_cast<__gm__ half*>(v1);
      v32.SetGlobalBuffer(v35, v34);
      AscendC::DataCopy(v31, v32, v29);
      v11.EnQue(v31);
      AscendC::LocalTensor<half> v36 = v11.DeQue<half>();
      v10.InitBuffer(v20, v30);
      uint32_t v37 = v29 * v16;
      uint32_t v38 = v37 * c2_idx;
      v10.InitBuffer(v19, v38);
      AscendC::LocalTensor<half> v39 = v19.Get<half>();
      int32_t v40 = static_cast<int32_t>(v29);
      int32_t v41 = static_cast<int32_t>(v16);
      v10.InitBuffer(v18, v38);
      AscendC::LocalTensor<half> v42 = v18.Get<half>();
      AscendC::Broadcast<half, half, 2>(v42, v36, reinterpret_cast<uint64_t>(v40), reinterpret_cast<uint64_t>(v40));
      v10.InitBuffer(v17, v38);
      AscendC::LocalTensor<half> v43 = v17.Get<half>();
      AscendC::GlobalTensor<half> v44;
      uint32_t v45 = v33 * v16;
      int32_t v46 = static_cast<int32_t>(v45);
      __gm__ half* v47 = reinterpret_cast<__gm__ half*>(v2);
      v44.SetGlobalBuffer(v47, v46);
      AscendC::DataCopy(v43, v44, v37);
      AscendC::Add(v39, v42, v43, v37);
      AscendC::Add(v39, v39, v39, v37);
      AscendC::LocalTensor<half> v48 = v12.AllocTensor<half>();
      AscendC::ReduceSum<AscendC::ReduceLayout::AR>(v48, v39);
      v12.EnQue(v48);
      AscendC::LocalTensor<half> v49 = v12.DeQue<half>();
      AscendC::GlobalTensor<half> v50;
      __gm__ half* v51 = reinterpret_cast<__gm__ half*>(v4);
      v50.SetGlobalBuffer(v51, v34);
      AscendC::DataCopy(v50, v49, v29);
      v12.FreeTensor(v49);
      v11.FreeTensor(v36);
    }
  }
  return;
}

