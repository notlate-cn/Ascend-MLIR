// RUN: afir-translate -mlir-to-cann %S/cann-translate-gather-input.mlir | FileCheck %s

// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(v2) + c0_i32);
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v1));
// CHECK-NOT: SetGlobalBuffer(v1)
// CHECK-NOT: = v46(
// CHECK: AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_idx32_tbuf_0;
// CHECK: InitBuffer(_afir_idx32_tbuf_0, (uint32_t)v43 * sizeof(uint32_t));
// CHECK: AscendC::LocalTensor<uint32_t> _afir_idx32_0 = _afir_idx32_tbuf_0.Get<uint32_t>();
// MTE2->S barrier so the scalar GetValue sees the DataCopy'd indices on real HW.
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>(v43); _afir_i++)
// CHECK: _afir_idx32_0.SetValue(_afir_i, static_cast<uint32_t>(v39.GetValue(_afir_i)) * 2);
// S->V barrier so the vector Gather sees the scalar-written byte offsets.
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: AscendC::GlobalTensor<half> _afir_gt;
// CHECK: _afir_gt.SetGlobalBuffer(
// CHECK-SAME: GetPhyAddr(
// CHECK: AscendC::DataCopy(
// CHECK-SAME: _afir_gt
// CHECK: uint32_t _afir_gather_count = static_cast<uint32_t>(v43);
// CHECK: for (uint32_t _afir_off = 0; _afir_off < _afir_gather_count; _afir_off += 128)
// CHECK: AscendC::Gather(
// CHECK-SAME: _afir_idx32_0[_afir_off]
// CHECK: AscendC::PipeBarrier<PIPE_V>();
// CHECK-NOT: v39
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v4) +
