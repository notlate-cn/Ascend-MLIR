// include/Runtime/AclnnBackend.h
// Reads a network.mlir (output of afir-opt) and emits network_host.cpp.
#pragma once

#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

struct AclnnBackendConfig {
  std::string networkMlirPath;  // input:  path to network.mlir
  std::string outputCppPath;    // output: path to network_host.cpp
};

class AclnnBackend {
public:
  llvm::Error generate(const AclnnBackendConfig &cfg);
};

} // namespace mlir::runtime
