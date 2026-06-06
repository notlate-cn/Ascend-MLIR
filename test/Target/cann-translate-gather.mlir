// RUN: ascend-mlir-translate -mlir-to-cann %S/cann-translate-gather-input.mlir | FileCheck %s

// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(v2) + c0_i32);
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v1));
// CHECK-NOT: SetGlobalBuffer(v1)
// CHECK-NOT: = v46(
// CHECK-NOT: _ascend_idx32
// CHECK: AscendC::GlobalTensor<half> _ascend_gt;
// CHECK: _ascend_gt.SetGlobalBuffer(
// CHECK-SAME: GetPhyAddr(
// CHECK: if ((_ascend_count * sizeof(half)) % 32u == 0u)
// CHECK: AscendC::DataCopy(
// CHECK-SAME: _ascend_gt
// CHECK: } else {
// CHECK: for (uint32_t _ascend_i = 0; _ascend_i < _ascend_count; ++_ascend_i)
// CHECK: {{v[0-9]+}}.SetValue(_ascend_i, _ascend_gt.GetValue(_ascend_i));
// CHECK: uint32_t _ascend_gather_count = static_cast<uint32_t>({{v[0-9]+}});
// CHECK: uint32_t _ascend_gather_padded_count = _ascend_gather_count == 0u ? 0u : ((_ascend_gather_count + 127u) / 128u) * 128u;
// CHECK: SetSize(_ascend_gather_padded_count);
// CHECK-NOT: $4
// CHECK: SetSize((uint32_t)
// CHECK-NOT: AscendC::Gather(
// CHECK: for (uint32_t _ascend_i = 0; _ascend_i < _ascend_gather_count; ++_ascend_i)
// CHECK: uint32_t _ascend_elem_offset = (static_cast<uint32_t>(
// CHECK-SAME: / 2u) + static_cast<uint32_t>(
// CHECK-SAME: .GetValue(_ascend_i));
// CHECK: SetValue(_ascend_i,
// CHECK-SAME: GetValue(_ascend_elem_offset)
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK-NOT: v39
// CHECK: AscendC::Max(
// CHECK: AscendC::LocalTensor<half> [[BIAS:v[0-9]+]] =
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v3) + c0_i32);
// CHECK: AscendC::DataCopy([[BIAS]],
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK: for (uint32_t _ascend_i = 0; _ascend_i < static_cast<uint32_t>(
// CHECK: _ascend_r = [[BIAS]].GetValue(_ascend_i)
// CHECK: _ascend_l = {{.*}}.GetValue(_ascend_i)
// CHECK: SetValue(_ascend_i, static_cast<half>(static_cast<float>(_ascend_l) + static_cast<float>(_ascend_r)));
// CHECK: SetSize((uint32_t)
// CHECK: AscendC::PipeBarrier<PIPE_ALL>();
// CHECK-NOT: AscendC::Add(
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v4) +
// CHECK: if ((_ascend_count * sizeof(half)) % 32u == 0u)
// CHECK: AscendC::DataCopy(
// CHECK-SAME: _ascend_count
// CHECK: } else {
// CHECK: {{v[0-9]+}}.SetValue(_ascend_i, {{v[0-9]+}}.GetValue(_ascend_i));
