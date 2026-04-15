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

#include <fstream>
#include <initializer_list>

namespace mlir::runtime {

namespace {

static constexpr const char *kStageAnalyzeSource = "analyze source";

static llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s",
                                   path.str().c_str());
  os << content.str();
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to write file: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

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

static std::string joinDefinitions(llvm::ArrayRef<std::string> defs) {
  std::string out;
  for (size_t i = 0; i < defs.size(); ++i) {
    if (i)
      out.push_back(';');
    out += defs[i];
  }
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

  auto compatOr = loadLegacyMixCompileContract(
      layout, sourcePath, cfg.kernelName, cfg.socVersion, *analyzed);
  if (!compatOr)
    return compatOr.takeError();

  MixAnalyzedKernel deviceAnalyzed = *analyzed;
  deviceAnalyzed.aicDefines = compatOr->aicDefinitions;
  deviceAnalyzed.aivDefines = compatOr->aivDefinitions;

  auto buildOr = executeLegacyMixBinaryBuild(*compatOr, sourcePath,
                                             cfg.kernelName, cfg.socVersion);
  if (!buildOr)
    return buildOr.takeError();

  auto abiOr = loadLegacyMixRuntimeAbi(
      cfg.cannMlirPath ? llvm::StringRef(*cfg.cannMlirPath) : llvm::StringRef(),
      cfg.npyDir ? llvm::StringRef(*cfg.npyDir) : llvm::StringRef(),
      buildOr->runtimeKernelName);
  if (!abiOr)
    return abiOr.takeError();
  MixAbiMetadata abi = std::move(*abiOr);

  auto tilingOr = executeLegacyMixTilingStage(layout, buildOr->runtimeKernelName,
                                              cfg.socVersion, abi);
  if (!tilingOr)
    return tilingOr.takeError();
  abi.blockDim = tilingOr->blockDim;

  auto metadataPathOr = writeLegacyMixCompileMetadataFile(
      layout.metadataPath, buildOr->runtimeKernelName, cfg.socVersion,
      "mix_1c1v", compatOr->generatedSourcePath, deviceAnalyzed.aicDefines,
      deviceAnalyzed.aivDefines, layout.mergedDeviceObj, layout.kernelSoPath,
      tilingOr->tilingArtifactPath, tilingOr->launchInfoPath, abi,
      tilingOr->usedLegacyRunner);
  if (!metadataPathOr)
    return metadataPathOr.takeError();

  if (auto err = writeTextFile(
          layout.analysisPath,
          std::string("kernel_name=") + buildOr->runtimeKernelName + "\n" +
              std::string("requested_kernel_name=") + analyzed->kernelName +
              "\n" +
              std::string("soc_version=") + analyzed->socVersion + "\n" +
              std::string("source_path=") + sourcePath.str().str() + "\n" +
              (buildOr->hostSourcePath.empty()
                   ? std::string{}
                   : std::string("host_source_path=") +
                         buildOr->hostSourcePath + "\n") +
              std::string("generated_source_path=") +
              compatOr->generatedSourcePath + "\n" +
              std::string("aic_definitions=") +
              joinDefinitions(deviceAnalyzed.aicDefines) + "\n" +
              std::string("aiv_definitions=") +
              joinDefinitions(deviceAnalyzed.aivDefines) + "\n" +
              std::string("aic_object=") + layout.aicObj + "\n" +
              std::string("aiv_object=") + layout.aivObj + "\n" +
              std::string("aic_reloc_object=") + layout.aicRelocObj + "\n" +
              std::string("aiv_reloc_object=") + layout.aivRelocObj + "\n" +
              std::string("device_object=") + layout.mergedDeviceObj + "\n"))
    return err;

  MixLegacyDebugManifestInputs debugInputs;
  debugInputs.analyzed = &*analyzed;
  debugInputs.abi = &abi;
  debugInputs.runtimeKernelName = buildOr->runtimeKernelName;
  debugInputs.sourcePath = sourcePath.str().str();
  debugInputs.hostSourcePath = buildOr->hostSourcePath;
  debugInputs.preprocessCompileCommandsPath =
      buildOr->preprocessCompileCommandsPath;
  debugInputs.preprocessCommand = buildOr->preprocessCommand;
  debugInputs.preprocessGeneratedDir = buildOr->preprocessGeneratedDir;
  debugInputs.generatedSourcePath = compatOr->generatedSourcePath;
  debugInputs.aicDefinitions = joinDefinitions(deviceAnalyzed.aicDefines);
  debugInputs.aivDefinitions = joinDefinitions(deviceAnalyzed.aivDefines);
  debugInputs.workDir = layout.workDir;
  debugInputs.objectDir = layout.objectDir;
  debugInputs.outDir = layout.outDir;
  debugInputs.mergeDir = layout.mergeDir;
  debugInputs.launcherHeaderDir = layout.outIncludeDir;
  debugInputs.hostStubSourcePath = buildOr->hostStubSourcePath;
  debugInputs.hostStubObjectPath = layout.hostStubObjectPath;
  debugInputs.kernelSoPath = layout.kernelSoPath;
  debugInputs.mixFlagPath = layout.mixFlagPath;
  debugInputs.runnerSourcePath = tilingOr->runnerSourcePath;
  debugInputs.runnerBinaryPath = tilingOr->runnerBinaryPath;
  debugInputs.aicObj = layout.aicObj;
  debugInputs.aivObj = layout.aivObj;
  debugInputs.aicRelocObj = layout.aicRelocObj;
  debugInputs.aivRelocObj = layout.aivRelocObj;
  debugInputs.mergedDeviceObj = layout.mergedDeviceObj;
  debugInputs.aicCompileCmd = buildOr->aicCompileCommand;
  debugInputs.aivCompileCmd = buildOr->aivCompileCommand;
  debugInputs.aicRelocCmd = buildOr->aicRelocCommand;
  debugInputs.aivRelocCmd = buildOr->aivRelocCommand;
  debugInputs.mergeCmd = buildOr->mergeCommand;
  debugInputs.hostCompileCmd = buildOr->hostCompileCommand;
  debugInputs.hostBishengObjectPath = buildOr->hostBishengObjectPath;
  debugInputs.hostBishengCmd = buildOr->hostBishengCommand;
  debugInputs.hostObjectDir = buildOr->hostObjectDir;
  debugInputs.packCmd = buildOr->packCommand;
  debugInputs.linkCmd = buildOr->hostLinkCommand;
  debugInputs.recompileCmd = buildOr->recompileCommand;
  debugInputs.runnerCompileCmd = tilingOr->runnerCompileCommand;
  debugInputs.metadataPath = *metadataPathOr;
  debugInputs.manifestPath = layout.manifestPath;
  if (auto err = writeLegacyMixDebugManifest(debugInputs))
    return err;

  MixArtifact artifact;
  artifact.kernel_name = buildOr->runtimeKernelName;
  artifact.soc_version = analyzed->socVersion;
  artifact.work_dir = layout.workDir;
  artifact.build_dir = layout.objectDir;
  artifact.install_dir = layout.outDir;
  artifact.kernel_so_path = layout.kernelSoPath;
  artifact.launcher_header_dir = layout.outIncludeDir;
  artifact.host_runner_path = tilingOr->runnerBinaryPath;
  artifact.host_stub_source_path = buildOr->hostStubSourcePath;
  artifact.device_object_path = layout.mergedDeviceObj;
  artifact.manifest_path = layout.manifestPath;
  artifact.metadata_path = *metadataPathOr;
  return artifact;
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
