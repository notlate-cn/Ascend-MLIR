#include "Runtime/Support/ToolDiscovery.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cstdlib>

namespace mlir::runtime {

std::string resolveSiblingToolPathForExecutable(llvm::StringRef executablePath,
                                                llvm::StringRef toolName) {
  if (executablePath.empty() || toolName.empty())
    return {};

  llvm::SmallString<256> toolPath(executablePath);
  llvm::sys::fs::make_absolute(toolPath);
  llvm::SmallString<256> parentPath =
      llvm::sys::path::parent_path(llvm::StringRef(toolPath));
  toolPath = parentPath;
  llvm::sys::path::append(toolPath, toolName);
  return toolPath.str().str();
}

void configureSiblingToolPathEnv(llvm::StringRef envName,
                                 llvm::StringRef executablePath,
                                 llvm::StringRef toolName) {
  if (envName.empty())
    return;
  const std::string env = envName.str();
  if (std::getenv(env.c_str()))
    return;

  const std::string toolPath =
      resolveSiblingToolPathForExecutable(executablePath, toolName);
  if (toolPath.empty())
    return;
  setenv(env.c_str(), toolPath.c_str(), /*overwrite=*/0);
}

} // namespace mlir::runtime
