#include "Runtime/ArtifactCompiler.h"

#include "Runtime/Compiler.h"
#include "Runtime/MixDirectBackend.h"
#include "Runtime/PathUtils.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

namespace {

static std::string defaultCompilerArch(KernelKind kind) {
  switch (kind) {
  case KernelKind::Cube:
    return "dav-c220-cube";
  case KernelKind::Vec:
  case KernelKind::Mix:
    return "dav-c220-vec";
  }
  return "dav-c220-vec";
}

} // namespace

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

  CompilerConfig cfg;
  cfg.soc_version = resolvedSoc;
  cfg.arch = defaultCompilerArch(req.kernelKind);
  cfg.kernel_type = req.kernelKind == KernelKind::Cube ? "cube" : "vec";

  Compiler compiler(cfg);
  auto binaryOr = compiler.Compile(req.kernelSource, req.outputDir,
                                   req.kernelName);
  if (!binaryOr)
    return binaryOr.takeError();

  return normalizeCompiledArtifact(*binaryOr, req.kernelName, req.kernelKind,
                                   resolvedSoc, req.outputDir);
}

} // namespace mlir::runtime
