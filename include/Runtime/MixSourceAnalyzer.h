#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::runtime {

struct MixAnalyzedKernel {
  std::string kernelName;
  std::string socVersion;
  std::string launcherSymbol;
  std::string aicEntry;
  std::string aivEntry;
  std::vector<std::string> commonFlags;
  std::vector<std::string> aicDefines;
  std::vector<std::string> aivDefines;
};

llvm::Expected<MixAnalyzedKernel>
analyzeMixKernel(llvm::StringRef kernelPath, llvm::StringRef kernelName,
                 llvm::StringRef socVersion);

} // namespace mlir::runtime
