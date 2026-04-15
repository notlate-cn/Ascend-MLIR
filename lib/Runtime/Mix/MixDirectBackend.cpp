#include "Runtime/MixDirectBackend.h"

#include "Runtime/MixAbi.h"
#include "Runtime/Mix/MixLegacyCompileCompat.h"
#include "Runtime/MixSourceAnalyzer.h"
#include "Runtime/Support/PathUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>

namespace mlir::runtime {

namespace {

static constexpr const char *kStageAnalyzeSource = "analyze source";

static std::string makeStageContext(
    std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>> fields) {
  std::string out;
  llvm::raw_string_ostream os(out);
  bool first = true;
  for (const auto &field : fields) {
    if (field.second.empty())
      continue;
    if (!first)
      os << ", ";
    first = false;
    os << field.first << "=" << field.second;
  }
  os.flush();
  return out;
}

} // namespace

llvm::Expected<MixArtifact>
MixDirectBackend::compile(const MixDirectCompileConfig &cfg) {
  if (cfg.outputDir.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires an output directory");
  if (cfg.kernelSrc.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires a kernel source path");
  if (cfg.kernelName.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires a kernel name");
  if (auto ascendHomeOr = requireAscendHome(); !ascendHomeOr)
    return ascendHomeOr.takeError();

  auto layoutOr = buildLegacyMixCompileLayout(cfg.outputDir, cfg.kernelName);
  if (!layoutOr)
    return layoutOr.takeError();
  const MixCompileLayout &layout = *layoutOr;

  const std::string analyzeContext = makeStageContext({
      {"kernel", cfg.kernelName},
      {"source", cfg.kernelSrc},
      {"soc_version", cfg.socVersion},
  });
  llvm::SmallString<256> sourcePath(cfg.kernelSrc);
  if (auto ec = llvm::sys::fs::make_absolute(sourcePath))
    return llvm::createStringError(
        ec, "[%s] cannot resolve kernel source path: %s (inputs: %s)",
        kStageAnalyzeSource, cfg.kernelSrc.c_str(), analyzeContext.c_str());
  if (!llvm::sys::fs::exists(sourcePath))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] kernel source file not found: %s (inputs: %s)",
        kStageAnalyzeSource, sourcePath.c_str(), analyzeContext.c_str());

  auto analyzed = analyzeMixKernel(sourcePath, cfg.kernelName, cfg.socVersion);
  if (!analyzed) {
    const std::string analysisError = llvm::toString(analyzed.takeError());
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "[%s] inputs: %s: %s",
                                   kStageAnalyzeSource,
                                   analyzeContext.c_str(),
                                   analysisError.c_str());
  }

  auto compileOr = executeLegacyMixCompilePipeline(
      layout, sourcePath, cfg.kernelName,
      cfg.cannMlirPath ? llvm::StringRef(*cfg.cannMlirPath) : llvm::StringRef(),
      cfg.npyDir ? llvm::StringRef(*cfg.npyDir) : llvm::StringRef(),
      cfg.socVersion, *analyzed);
  if (!compileOr)
    return compileOr.takeError();
  const MixLegacyCompileOutputs &compile = *compileOr;

  return finalizeLegacyMixArtifact(layout, sourcePath, *analyzed, compile);
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
  normalized.packedSharedObjectPath = artifact.kernel_so_path;
  normalized.manifestPath = artifact.manifest_path;
  normalized.metadataPath = artifact.metadata_path;
  return normalized;
}

} // namespace mlir::runtime
