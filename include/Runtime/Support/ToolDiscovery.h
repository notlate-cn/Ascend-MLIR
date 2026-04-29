#pragma once

#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::runtime {

std::string resolveSiblingToolPathForExecutable(llvm::StringRef executablePath,
                                                llvm::StringRef toolName);

void configureSiblingToolPathEnv(llvm::StringRef envName,
                                 llvm::StringRef executablePath,
                                 llvm::StringRef toolName);

} // namespace mlir::runtime
