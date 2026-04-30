#include "Runtime/Execution/DynamicLibraryArtifactEnv.h"

#include "Runtime/PathUtils.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <string>

namespace mlir::runtime {

namespace {

void prependEnvPath(const char *name, const std::string &prefix) {
  if (prefix.empty())
    return;
  const char *current = std::getenv(name);
  std::string value = prefix;
  if (current && *current) {
    value.push_back(':');
    value += current;
  }
  ::setenv(name, value.c_str(), 1);
}

} // namespace

llvm::Error
configureDynamicLibraryArtifactSimulationEnv(const KernelArtifact &artifact) {
  const std::string ascendHome = findAscendHome();
  if (ascendHome.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Ascend toolkit root is not configured; set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME");

  const std::string resolvedSoc =
      resolveSocVersion(artifact.socVersion, "Ascend910B1");
  const std::string ascendLib64 = findAscendLib64Dir(ascendHome);
  const std::string simLibDir =
      findAscendSimulatorLibDir(ascendHome, resolvedSoc);
  auto davSimLibDirOr = requireAscendDavSimulatorLibDir(ascendHome);
  if (!davSimLibDirOr)
    return davSimLibDirOr.takeError();
  const std::string deviceLibDir = findAscendDeviceLibDir(ascendHome);

  prependEnvPath("LD_LIBRARY_PATH", artifact.artifactRoot + "/out");
  prependEnvPath("LD_LIBRARY_PATH", ascendLib64);
  prependEnvPath("LD_LIBRARY_PATH", simLibDir);
  prependEnvPath("LD_LIBRARY_PATH", *davSimLibDirOr);
  prependEnvPath("LD_LIBRARY_PATH", deviceLibDir);

  return llvm::Error::success();
}

} // namespace mlir::runtime
