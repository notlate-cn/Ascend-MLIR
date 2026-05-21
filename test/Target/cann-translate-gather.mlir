// RUN: afir-translate -mlir-to-cann %S/cann-translate-gather-input.mlir | FileCheck %s

// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(v2) + c0_i32);
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v1));
// CHECK-NOT: SetGlobalBuffer(v1)
// CHECK-NOT: = v46(
// CHECK: AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_idx32_tbuf_0;
// CHECK: uint32_t _afir_idx32_count = static_cast<uint32_t>([[INDEX_COUNT:v[0-9]+]]);
// CHECK: uint32_t _afir_idx32_padded_count = _afir_idx32_count == 0u ? 0u : ((_afir_idx32_count + 127u) / 128u) * 128u;
// CHECK: uint32_t _afir_idx32_bytes = _afir_idx32_padded_count * sizeof(uint32_t);
// CHECK: uint32_t _afir_idx32_aligned_bytes = _afir_idx32_bytes == 0u ? 0u : ((_afir_idx32_bytes + 31u) / 32u) * 32u;
// CHECK: if (_afir_idx32_bytes != 0u && _afir_idx32_aligned_bytes < 32u)
// CHECK: _afir_idx32_aligned_bytes = 32u;
// CHECK: InitBuffer(_afir_idx32_tbuf_0, _afir_idx32_aligned_bytes);
// CHECK: AscendC::LocalTensor<uint32_t> _afir_idx32_0 = _afir_idx32_tbuf_0.Get<uint32_t>();
// CHECK: SetSize(_afir_idx32_count);
// CHECK: for (uint32_t _afir_i = 0; _afir_i < _afir_idx32_count; _afir_i++)
// CHECK: _afir_idx32_0.SetValue(_afir_i, static_cast<uint32_t>({{v[0-9]+}}.GetValue(_afir_i)) * 2u);
// CHECK: for (uint32_t _afir_i = _afir_idx32_count; _afir_i < _afir_idx32_padded_count; _afir_i++)
// CHECK: _afir_idx32_0.SetValue(_afir_i, 0u);
// CHECK: _afir_idx32_0.SetSize(_afir_idx32_padded_count);
// CHECK: AscendC::PipeBarrier<PIPE_V>();
// CHECK: AscendC::GlobalTensor<half> _afir_gt;
// CHECK: _afir_gt.SetGlobalBuffer(
// CHECK-SAME: GetPhyAddr(
// CHECK: if ((_afir_count * sizeof(half)) % 32u == 0u)
// CHECK: AscendC::DataCopy(
// CHECK-SAME: _afir_gt
// CHECK: } else {
// CHECK: for (uint32_t _afir_i = 0; _afir_i < _afir_count; ++_afir_i)
// CHECK: {{v[0-9]+}}.SetValue(_afir_i, _afir_gt.GetValue(_afir_i));
// CHECK: uint32_t _afir_gather_count = static_cast<uint32_t>({{v[0-9]+}});
// CHECK: uint32_t _afir_gather_padded_count = _afir_gather_count == 0u ? 0u : ((_afir_gather_count + 127u) / 128u) * 128u;
// CHECK: SetSize(_afir_gather_padded_count);
// CHECK-NOT: $4
// CHECK: SetSize((uint32_t)
// CHECK: for (uint32_t _afir_off = 0; _afir_off < _afir_gather_count; _afir_off += 128)
// CHECK: AscendC::Gather(
// CHECK-SAME: _afir_idx32_0[_afir_off]
// CHECK: AscendC::PipeBarrier<PIPE_V>();
// CHECK-NOT: v39
// CHECK: AscendC::Max(
// CHECK: AscendC::LocalTensor<half> [[BIAS:v[0-9]+]] =
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v3) + c0_i32);
// CHECK: AscendC::DataCopy([[BIAS]],
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>(
// CHECK: _afir_r = [[BIAS]].GetValue(_afir_i)
// CHECK: _afir_l = {{.*}}.GetValue(_afir_i)
// CHECK: SetValue(_afir_i, static_cast<half>(static_cast<float>(_afir_l) + static_cast<float>(_afir_r)));
// CHECK: SetSize((uint32_t)
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK-NOT: AscendC::Add(
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v4) +
// CHECK: if ((_afir_count * sizeof(half)) % 32u == 0u)
// CHECK: AscendC::DataCopy(
// CHECK-SAME: _afir_count
// CHECK: } else {
// CHECK: {{v[0-9]+}}.SetValue(_afir_i, {{v[0-9]+}}.GetValue(_afir_i));
