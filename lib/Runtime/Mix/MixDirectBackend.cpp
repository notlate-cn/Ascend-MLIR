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

  auto compileOr = executeLegacyMixCompilePipeline(
      layout, sourcePath, cfg.kernelName,
      cfg.cannMlirPath ? llvm::StringRef(*cfg.cannMlirPath) : llvm::StringRef(),
      cfg.npyDir ? llvm::StringRef(*cfg.npyDir) : llvm::StringRef(),
      cfg.socVersion, *analyzed);
  if (!compileOr)
    return compileOr.takeError();
  const MixLegacyCompileOutputs &compile = *compileOr;

  if (auto err = writeTextFile(
          layout.analysisPath,
          std::string("kernel_name=") + compile.build.runtimeKernelName + "\n" +
              std::string("requested_kernel_name=") + analyzed->kernelName +
              "\n" +
              std::string("soc_version=") + analyzed->socVersion + "\n" +
              std::string("source_path=") + sourcePath.str().str() + "\n" +
              (compile.build.hostSourcePath.empty()
                   ? std::string{}
                   : std::string("host_source_path=") +
                         compile.build.hostSourcePath + "\n") +
              std::string("generated_source_path=") +
              compile.contract.generatedSourcePath + "\n" +
              std::string("aic_definitions=") +
              joinDefinitions(compile.contract.aicDefinitions) + "\n" +
              std::string("aiv_definitions=") +
              joinDefinitions(compile.contract.aivDefinitions) + "\n" +
              std::string("aic_object=") + layout.aicObj + "\n" +
              std::string("aiv_object=") + layout.aivObj + "\n" +
              std::string("aic_reloc_object=") + layout.aicRelocObj + "\n" +
              std::string("aiv_reloc_object=") + layout.aivRelocObj + "\n" +
              std::string("device_object=") + layout.mergedDeviceObj + "\n"))
    return err;

  MixLegacyDebugManifestInputs debugInputs;
  debugInputs.analyzed = &*analyzed;
  debugInputs.abi = &compile.abi;
  debugInputs.runtimeKernelName = compile.build.runtimeKernelName;
  debugInputs.sourcePath = sourcePath.str().str();
  debugInputs.hostSourcePath = compile.build.hostSourcePath;
  debugInputs.preprocessCompileCommandsPath =
      compile.build.preprocessCompileCommandsPath;
  debugInputs.preprocessCommand = compile.build.preprocessCommand;
  debugInputs.preprocessGeneratedDir = compile.build.preprocessGeneratedDir;
  debugInputs.generatedSourcePath = compile.contract.generatedSourcePath;
  debugInputs.aicDefinitions = joinDefinitions(compile.contract.aicDefinitions);
  debugInputs.aivDefinitions = joinDefinitions(compile.contract.aivDefinitions);
  debugInputs.workDir = layout.workDir;
  debugInputs.objectDir = layout.objectDir;
  debugInputs.outDir = layout.outDir;
  debugInputs.mergeDir = layout.mergeDir;
  debugInputs.launcherHeaderDir = layout.outIncludeDir;
  debugInputs.hostStubSourcePath = compile.build.hostStubSourcePath;
  debugInputs.hostStubObjectPath = layout.hostStubObjectPath;
  debugInputs.kernelSoPath = layout.kernelSoPath;
  debugInputs.mixFlagPath = layout.mixFlagPath;
  debugInputs.runnerSourcePath = compile.tiling.runnerSourcePath;
  debugInputs.runnerBinaryPath = compile.tiling.runnerBinaryPath;
  debugInputs.aicObj = layout.aicObj;
  debugInputs.aivObj = layout.aivObj;
  debugInputs.aicRelocObj = layout.aicRelocObj;
  debugInputs.aivRelocObj = layout.aivRelocObj;
  debugInputs.mergedDeviceObj = layout.mergedDeviceObj;
  debugInputs.aicCompileCmd = compile.build.aicCompileCommand;
  debugInputs.aivCompileCmd = compile.build.aivCompileCommand;
  debugInputs.aicRelocCmd = compile.build.aicRelocCommand;
  debugInputs.aivRelocCmd = compile.build.aivRelocCommand;
  debugInputs.mergeCmd = compile.build.mergeCommand;
  debugInputs.hostCompileCmd = compile.build.hostCompileCommand;
  debugInputs.hostBishengObjectPath = compile.build.hostBishengObjectPath;
  debugInputs.hostBishengCmd = compile.build.hostBishengCommand;
  debugInputs.hostObjectDir = compile.build.hostObjectDir;
  debugInputs.packCmd = compile.build.packCommand;
  debugInputs.linkCmd = compile.build.hostLinkCommand;
  debugInputs.recompileCmd = compile.build.recompileCommand;
  debugInputs.runnerCompileCmd = compile.tiling.runnerCompileCommand;
  debugInputs.metadataPath = compile.metadataPath;
  debugInputs.manifestPath = layout.manifestPath;
  if (auto err = writeLegacyMixDebugManifest(debugInputs))
    return err;

  MixArtifact artifact;
  artifact.kernel_name = compile.build.runtimeKernelName;
  artifact.soc_version = analyzed->socVersion;
  artifact.work_dir = layout.workDir;
  artifact.build_dir = layout.objectDir;
  artifact.install_dir = layout.outDir;
  artifact.kernel_so_path = layout.kernelSoPath;
  artifact.launcher_header_dir = layout.outIncludeDir;
  artifact.host_runner_path = compile.tiling.runnerBinaryPath;
  artifact.host_stub_source_path = compile.build.hostStubSourcePath;
  artifact.device_object_path = layout.mergedDeviceObj;
  artifact.manifest_path = layout.manifestPath;
  artifact.metadata_path = compile.metadataPath;
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
