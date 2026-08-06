// RUN: ascend-mlir-translate -mlir-to-cann %s --artifact-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST

// CPP: extern "C" __global__ __aicore__ void add_kernel
// MANIFEST: "kernelName": "add_kernel"
// MANIFEST: "kernel_entries": [
// HOST: int32_t add_kernel_GetTilingSize(void)
// HOST: int32_t add_kernel_GetTiling(const int64_t* shape_args, int32_t shape_count, void* tiling_out)

module {
  func.func @add_kernel(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 2 : i32,
      cann.workspace_size_bytes = 0 : i64} {
    func.return
  }
}
