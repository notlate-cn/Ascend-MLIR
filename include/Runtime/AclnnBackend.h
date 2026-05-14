// include/Runtime/AclnnBackend.h
// Reads a network.mlir (output of afir-opt) and emits network_host.cpp.
#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

struct AclnnBackendConfig {
  std::string networkMlirPath;     // input:  path to network.mlir
  std::string outputCppPath;       // output: path to network_host.cpp
  std::string tilingsPath;         // input:  path to tilings JSON (per-kernel best params)
  std::string kernelBinariesDir;   // input:  dir of compiled AscendC kernel artifacts
  // Family-id → variant kernel name, built from tilings JSON keys at
  // generate() time. When set, emitAscendCLaunch substitutes the variant
  // suffix the autotuner picked (e.g. "kernel_group0" → "kernel_group0__v1").
  // When empty, emit defaults to "<family>__v0".
  llvm::StringMap<std::string> variantOverrides;
};

class AclnnBackend {
public:
  llvm::Error generate(const AclnnBackendConfig &cfg);
};

} // namespace mlir::runtime
