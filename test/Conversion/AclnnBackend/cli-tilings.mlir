// RUN: rm -f %t.cpp && aclnn-backend --input %s --output %t.cpp \
// RUN:   --tilings %S/Inputs/tilings_default.json \
// RUN:   --kernel-binaries %S/Inputs/artifacts
// RUN: FileCheck %s < %t.cpp

// CHECK: // tilings_path: {{.*}}tilings_default.json
// CHECK: // kernel_binaries_dir: {{.*}}artifacts

module {
  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    return %x : tensor<8xf16>
  }
}
