// RUN: afir-opt --ascend-normalize --ascend-kernelize --ascend-schedule %s | FileCheck %s

module {
  func.func @empty() {
    return
  }
}

// CHECK: func.func @empty
