// RUN: afir-translate -mlir-to-cann %s --artifact-manifest-out=%t.manifest.json > %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST

// CPP: extern "C" __global__ __aicore__ void compat_kernel
// MANIFEST: "kernelName": "compat_kernel"

module {
  func.func @compat_kernel(
      %input: memref<?xf16>,
      %output: memref<?xf16>,
      %workspace: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["dim_arg0_0"]>
  ) attributes {
      ascendc.aicore,
      ascendc.global,
      cann.num_inputs = 1 : i32,
      cann.workspace_size_bytes = 0 : i64} {
    func.return
  }
}
