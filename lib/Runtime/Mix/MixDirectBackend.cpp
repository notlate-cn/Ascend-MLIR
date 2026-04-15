#include "Runtime/MixDirectBackend.h"

#include "Runtime/Mix/MixDirectCompilePipeline.h"
#include "llvm/Support/Path.h"

namespace mlir::runtime {

llvm::Expected<MixArtifact>
MixDirectBackend::compile(const MixDirectCompileConfig &cfg) {
  return executeMixDirectCompile(
      cfg.outputDir, cfg.kernelSrc, cfg.kernelName,
      cfg.cannMlirPath ? llvm::StringRef(*cfg.cannMlirPath) : llvm::StringRef(),
      cfg.npyDir ? llvm::StringRef(*cfg.npyDir) : llvm::StringRef(),
      cfg.socVersion);
}

KernelArtifact normalizeMixArtifact(const MixArtifact &artifact, KernelKind kind,
                                    MixResourceType mixResourceType) {
  KernelArtifact normalized;
  normalized.kernelName = artifact.kernel_name;
  normalized.kernelKind = kind;
  normalized.mixResourceType = mixResourceType;
  normalized.socVersion = artifact.soc_version;
  normalized.artifactRoot =
      llvm::sys::path::parent_path(artifact.work_dir).str();
  if (normalized.artifactRoot.empty())
    normalized.artifactRoot = artifact.install_dir;
  normalized.deviceBinaryPath = artifact.device_object_path.empty()
                                    ? artifact.kernel_so_path
                                    : artifact.device_object_path;
  normalized.sharedLibraryPath = artifact.kernel_so_path;
  normalized.manifestPath = artifact.manifest_path;
  normalized.metadataPath = artifact.metadata_path;
  return normalized;
}

} // namespace mlir::runtime
