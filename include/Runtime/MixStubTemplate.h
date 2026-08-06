#pragma once

#include <cstddef>
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

struct MixStubTemplateArgs {
  std::string kernelName;
  std::string targetName;
  std::string socVersion;
  std::string launcherSymbol;
  std::string launcherHeaderPath;
  std::string hostStubSourcePath;
  size_t mixLen = 0;
  size_t mixFileLen = 0;
};

llvm::Error writeMixStubTemplate(const MixStubTemplateArgs &args);

} // namespace mlir::runtime
