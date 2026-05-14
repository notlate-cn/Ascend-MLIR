// RUN: afir-translate -mlir-to-cann %S/cann-translate-gather-input.mlir | FileCheck %s

// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(v2) + c0_i32);
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v1));
// CHECK-NOT: SetGlobalBuffer(v1)
// CHECK-NOT: = v46(
// CHECK: AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_idx32_tbuf_0;
// CHECK: uint32_t _afir_idx32_bytes = (uint32_t)v43 * sizeof(uint32_t);
// CHECK: uint32_t _afir_idx32_aligned_bytes = _afir_idx32_bytes == 0 ? 0 : ((_afir_idx32_bytes + 31u) / 32u) * 32u;
// CHECK: if (_afir_idx32_bytes != 0u && _afir_idx32_aligned_bytes < 32u)
// CHECK: _afir_idx32_aligned_bytes = 32u;
// CHECK: InitBuffer(_afir_idx32_tbuf_0, _afir_idx32_aligned_bytes);
// CHECK: AscendC::LocalTensor<uint32_t> _afir_idx32_0 = _afir_idx32_tbuf_0.Get<uint32_t>();
// CHECK: SetSize((uint32_t)v43);
// CHECK: for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>(v43); _afir_i++)
// CHECK: _afir_idx32_0.SetValue(_afir_i, static_cast<uint32_t>(v39.GetValue(_afir_i)) * 2u);
// CHECK: _afir_idx32_0.SetSize((uint32_t)v43);
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
// CHECK: uint32_t _afir_gather_count = static_cast<uint32_t>(v43);
// CHECK: SetSize(_afir_gather_count);
// CHECK: SetSize((uint32_t)
// CHECK: for (uint32_t _afir_off = 0; _afir_off < _afir_gather_count; _afir_off += 128)
// CHECK: AscendC::Gather(
// CHECK-SAME: _afir_idx32_0[_afir_off]
// CHECK: AscendC::PipeBarrier<PIPE_V>();
// CHECK-NOT: v39
// CHECK: for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>(
// CHECK: _afir_r = {{.*}}.GetValue(_afir_i)
// CHECK: AscendC::Adds(
// CHECK: SetSize((uint32_t)
// CHECK-NOT: AscendC::Add(
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v4) +
// CHECK: if ((_afir_count * sizeof(half)) % 32u == 0u)
// CHECK: AscendC::DataCopy(
// CHECK-SAME: _afir_count
// CHECK: } else {
// CHECK: {{v[0-9]+}}.SetValue(_afir_i, {{v[0-9]+}}.GetValue(_afir_i));
