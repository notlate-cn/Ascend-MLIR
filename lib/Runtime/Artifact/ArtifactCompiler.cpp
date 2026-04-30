#include "Runtime/ArtifactCompiler.h"

#include "Runtime/Artifact/VecCubeArtifactBackend.h"
#include "Runtime/MixDirectBackend.h"
#include "Runtime/PathUtils.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

MixResourceType inferMixResourceTypeFromKernelKind(KernelKind kind) {
  switch (kind) {
  case KernelKind::Mix:
    return MixResourceType::Mix1C1V;
  case KernelKind::Vec:
  case KernelKind::Cube:
    return MixResourceType::Unknown;
  }
  return MixResourceType::Unknown;
}

llvm::Expected<KernelArtifact>
ArtifactCompiler::compile(const ArtifactCompileRequest &req) const {
  if (req.kernelSource.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel source is required");
  if (req.kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel name is required");
  if (req.outputDir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");

  const std::string resolvedSoc =
      resolveSocVersion(req.socVersion, "Ascend910B1");

  if (req.kernelKind == KernelKind::Mix) {
    // The mix backend compiles both cube and vec variants internally, so the
    // compatibility arch flag does not change this path.
    MixDirectCompileConfig cfg;
    cfg.kernelSrc = req.kernelSource;
    cfg.kernelName = req.kernelName;
    cfg.socVersion = resolvedSoc;
    cfg.outputDir = req.outputDir;
    cfg.cannMlirPath = req.cannMlirPath;
    cfg.npyDir = req.npyDir;

    MixDirectBackend backend;
    auto mixOr = backend.compile(cfg);
    if (!mixOr)
      return mixOr.takeError();
    KernelArtifact artifact = normalizeMixArtifact(*mixOr, req.kernelKind,
                                inferMixResourceTypeFromKernelKind(
                                    req.kernelKind));
    artifact.socVersion = resolvedSoc;
    return artifact;
  }

  VecCubeArtifactBackend backend;
  return backend.compile(req, resolvedSoc);
}

} // namespace mlir::runtime
