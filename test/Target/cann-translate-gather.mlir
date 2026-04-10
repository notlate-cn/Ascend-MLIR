// RUN: afir-translate -mlir-to-cann %S/cann-translate-gather-input.mlir | FileCheck %s

// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(v2) + c0_i32);
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v1));
// CHECK-NOT: SetGlobalBuffer(v1)
// CHECK-NOT: = v46(
// CHECK: AscendC::GlobalTensor<half> _afir_gt;
// CHECK: _afir_gt.SetGlobalBuffer(
// CHECK-SAME: GetPhyAddr(
// CHECK: AscendC::DataCopy(
// CHECK-SAME: _afir_gt
// CHECK: AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_idx32_tbuf;
// CHECK: v15.InitBuffer(_afir_idx32_tbuf, (uint32_t)v43 * sizeof(uint32_t));
// CHECK: AscendC::LocalTensor<uint32_t> _afir_idx32 = _afir_idx32_tbuf.Get<uint32_t>();
// CHECK: for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>(v43); _afir_i++)
// CHECK: _afir_idx32.SetValue(_afir_i, static_cast<uint32_t>(v39.GetValue(_afir_i)));
// CHECK: AscendC::Gather(
// CHECK-SAME: _afir_idx32
// CHECK-NOT: AscendC::Gather(
// CHECK-NOT: v39
// CHECK: SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v4) +
