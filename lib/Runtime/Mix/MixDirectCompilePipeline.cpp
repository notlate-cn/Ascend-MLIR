#include "Runtime/Mix/MixDirectCompilePipeline.h"
#include "MixDirectCompileInternal.h"

#include "Runtime/Mix/MixSourceAnalyzer.h"
#include "Runtime/Support/PathUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"


namespace mlir::runtime {

namespace {

static constexpr const char *kStageAnalyzeSource = "analyze source";

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

llvm::Error
writeMixDirectDebugManifest(const MixDirectDebugManifestInputs &inputs) {
  if (!inputs.analyzed || !inputs.abi)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "debug manifest requires analyzed kernel and ABI");

  std::string manifest;
  manifest += std::string("kernel_name=") + inputs.runtimeKernelName + "\n";
  manifest +=
      std::string("requested_kernel_name=") + inputs.analyzed->kernelName + "\n";
  manifest += std::string("soc_version=") + inputs.analyzed->socVersion + "\n";
  manifest += std::string("kernel_kind=mix\n");
  manifest += std::string("mix_resource_type=mix_1c1v\n");
  manifest += std::string("source_path=") + inputs.sourcePath + "\n";
  if (!inputs.hostSourcePath.empty())
    manifest += std::string("host_source_path=") + inputs.hostSourcePath + "\n";
  manifest += std::string("preprocess_compile_commands=") +
              inputs.preprocessCompileCommandsPath + "\n";
  manifest +=
      std::string("preprocess_command=") + inputs.preprocessCommand + "\n";
  manifest += std::string("preprocess_generated_dir=") +
              inputs.preprocessGeneratedDir + "\n";
  manifest +=
      std::string("generated_source_path=") + inputs.generatedSourcePath + "\n";
  manifest += std::string("aic_definitions=") + inputs.aicDefinitions + "\n";
  manifest += std::string("aiv_definitions=") + inputs.aivDefinitions + "\n";
  manifest += std::string("work_dir=") + inputs.workDir + "\n";
  manifest += std::string("build_dir=") + inputs.objectDir + "\n";
  manifest += std::string("install_dir=") + inputs.outDir + "\n";
  manifest += std::string("object_dir=") + inputs.objectDir + "\n";
  manifest += std::string("out_dir=") + inputs.outDir + "\n";
  if (inputs.metadataPath.empty()) {
    manifest += std::string("abi_kind=mix_gm_workspace_tiling\n");
    auto abiManifestOr = serializeMixAbiManifest(*inputs.abi);
    if (!abiManifestOr)
      return abiManifestOr.takeError();
    manifest += *abiManifestOr;
  }
  manifest += std::string("merge_obj_dir=") + inputs.mergeDir + "\n";
  manifest += std::string("launcher_header_dir=") + inputs.launcherHeaderDir +
              "\n";
  manifest += std::string("manifest_path=") + inputs.manifestPath + "\n";
  if (!inputs.metadataPath.empty())
    manifest += std::string("metadata_path=") + inputs.metadataPath + "\n";
  if (!inputs.manifestPath.empty()) {
    llvm::SmallString<256> timingPath(
        llvm::sys::path::parent_path(inputs.manifestPath));
    llvm::sys::path::append(timingPath, "compile_timing.json");
    manifest +=
        std::string("compile_timing_path=") + timingPath.str().str() + "\n";
  }
  if (!inputs.hostStubSourcePath.empty())
    manifest +=
        std::string("host_stub_source_path=") + inputs.hostStubSourcePath + "\n";
  manifest += std::string("host_stub_object_path=") + inputs.hostStubObjectPath +
              "\n";
  if (!inputs.hostObjectDir.empty())
    manifest += std::string("host_object_dir=") + inputs.hostObjectDir + "\n";
  manifest += std::string("kernel_so_path=") + inputs.kernelSoPath + "\n";
  manifest += std::string("mix_build_flag=") + inputs.mixFlagPath + "\n";
  manifest += std::string("aic_object=") + inputs.aicObj + "\n";
  manifest += std::string("aiv_object=") + inputs.aivObj + "\n";
  manifest += std::string("aic_reloc_object=") + inputs.aicRelocObj + "\n";
  manifest += std::string("aiv_reloc_object=") + inputs.aivRelocObj + "\n";
  manifest += std::string("bisheng_aic=") + inputs.aicCompileCmd + "\n";
  manifest += std::string("bisheng_aiv=") + inputs.aivCompileCmd + "\n";
  manifest += std::string("lld_reloc_aic=") + inputs.aicRelocCmd + "\n";
  manifest += std::string("lld_reloc_aiv=") + inputs.aivRelocCmd + "\n";
  manifest += std::string("lld_merge=") + inputs.mergeCmd + "\n";
  manifest += std::string("host_compile_cmd=") + inputs.hostCompileCmd + "\n";
  manifest += std::string("pack_cmd=") + inputs.packCmd + "\n";
  manifest += std::string("host_link_cmd=") + inputs.linkCmd + "\n";
  if (!inputs.tilingBackendKind.empty())
    manifest += std::string("tiling_backend=") + inputs.tilingBackendKind + "\n";
  if (!inputs.tilingStrategyName.empty())
    manifest +=
        std::string("tiling_strategy=") + inputs.tilingStrategyName + "\n";
  if (!inputs.tilingDebugNote.empty())
    manifest +=
        std::string("tiling_debug_note=") + inputs.tilingDebugNote + "\n";
  manifest += std::string("tiling_compile_command=") + inputs.runnerCompileCmd +
              "\n";
  manifest += std::string("tiling_helper_cmd=") +
              inputs.runnerCompileCmd + "\n";
  if (!inputs.mergedDeviceObj.empty())
    manifest +=
        std::string("device_object_path=") + inputs.mergedDeviceObj + "\n";
  return writeTextFile(inputs.manifestPath, manifest);
}

llvm::Expected<MixDirectCompileOutputs>
executeMixDirectCompilePipeline(const MixCompileLayout &layout,
                                llvm::StringRef sourcePath,
                                llvm::StringRef requestedKernelName,
                                llvm::StringRef cannMlirPath,
                                llvm::StringRef npyDir,
                                llvm::StringRef socVersion,
                                const MixAnalyzedKernel &analyzed) {
  MixDirectCompileOutputs outputs;
  {
    MixDirectStageTimer timer("preprocess_contract", outputs.timings);
    auto contractOr = loadMixDirectCompileContract(
        layout, sourcePath, requestedKernelName, socVersion, analyzed);
    if (!contractOr)
      return contractOr.takeError();
    outputs.contract = std::move(*contractOr);
  }

  auto buildOr = executeMixDirectBinaryBuild(outputs.contract, sourcePath,
                                             requestedKernelName, socVersion);
  if (!buildOr)
    return buildOr.takeError();
  outputs.build = std::move(*buildOr);
  outputs.timings.insert(outputs.timings.end(), outputs.build.timings.begin(),
                         outputs.build.timings.end());

  {
    MixDirectStageTimer timer("load_runtime_abi", outputs.timings);
    auto abiOr = loadMixDirectRuntimeAbi(cannMlirPath, npyDir,
                                         outputs.build.runtimeKernelName);
    if (!abiOr)
      return abiOr.takeError();
    outputs.abi = std::move(*abiOr);
  }

  auto tilingOr = executeMixDirectTilingStage(
      layout, outputs.build.runtimeKernelName, socVersion, outputs.abi);
  if (!tilingOr)
    return tilingOr.takeError();
  outputs.tiling = std::move(*tilingOr);
  outputs.timings.insert(outputs.timings.end(), outputs.tiling.timings.begin(),
                         outputs.tiling.timings.end());
  outputs.abi.blockDim = outputs.tiling.blockDim;

  {
    MixDirectStageTimer timer("write_metadata", outputs.timings);
    auto metadataPathOr = writeMixDirectCompileMetadataFile(
        layout.metadataPath, outputs.build.runtimeKernelName, socVersion,
        "mix_1c1v", outputs.contract.generatedSourcePath,
        outputs.contract.aicDefinitions, outputs.contract.aivDefinitions,
        layout.mergedDeviceObj, layout.kernelSoPath, outputs.tiling,
        outputs.abi);
    if (!metadataPathOr)
      return metadataPathOr.takeError();
    outputs.metadataPath = *metadataPathOr;
  }
  {
    MixDirectStageTimer timer("write_timing_file", outputs.timings);
    if (auto err = writeMixDirectTimingFile(layout.timingPath, outputs.timings))
      return std::move(err);
  }
  return outputs;
}

llvm::Expected<MixArtifact>
finalizeMixDirectArtifact(const MixCompileLayout &layout,
                          llvm::StringRef sourcePath,
                          const MixAnalyzedKernel &analyzed,
                          const MixDirectCompileOutputs &compile) {
  if (auto err = writeTextFile(
          layout.analysisPath,
          std::string("kernel_name=") + compile.build.runtimeKernelName + "\n" +
              std::string("requested_kernel_name=") + analyzed.kernelName +
              "\n" +
              std::string("soc_version=") + analyzed.socVersion + "\n" +
              std::string("source_path=") + sourcePath.str() + "\n" +
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
    return std::move(err);

  MixDirectDebugManifestInputs debugInputs;
  debugInputs.analyzed = &analyzed;
  debugInputs.abi = &compile.abi;
  debugInputs.runtimeKernelName = compile.build.runtimeKernelName;
  debugInputs.sourcePath = sourcePath.str();
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
  debugInputs.hostObjectDir = compile.build.hostObjectDir;
  debugInputs.packCmd = compile.build.packCommand;
  debugInputs.linkCmd = compile.build.hostLinkCommand;
  debugInputs.tilingBackendKind = compile.tiling.backendKind;
  debugInputs.tilingStrategyName = compile.tiling.strategyName;
  debugInputs.tilingDebugNote = compile.tiling.debugNote;
  debugInputs.runnerCompileCmd = compile.tiling.runnerCompileCommand;
  debugInputs.metadataPath = compile.metadataPath;
  debugInputs.manifestPath = layout.manifestPath;
  if (auto err = writeMixDirectDebugManifest(debugInputs))
    return std::move(err);

  MixArtifact artifact;
  artifact.kernel_name = compile.build.runtimeKernelName;
  artifact.soc_version = analyzed.socVersion;
  artifact.work_dir = layout.workDir;
  artifact.build_dir = layout.objectDir;
  artifact.install_dir = layout.outDir;
  artifact.kernel_so_path = layout.kernelSoPath;
  artifact.launcher_header_dir = layout.outIncludeDir;
  artifact.host_stub_source_path = compile.build.hostStubSourcePath;
  artifact.device_object_path = layout.mergedDeviceObj;
  artifact.manifest_path = layout.manifestPath;
  artifact.metadata_path = compile.metadataPath;
  return artifact;
}

llvm::Expected<MixArtifact>
executeMixDirectCompile(llvm::StringRef outputDir, llvm::StringRef kernelSrc,
                        llvm::StringRef kernelName,
                        llvm::StringRef cannMlirPath, llvm::StringRef npyDir,
                        llvm::StringRef socVersion) {
  if (outputDir.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires an output directory");
  if (kernelSrc.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires a kernel source path");
  if (kernelName.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RuntimeMix direct backend requires a kernel name");
  if (auto ascendHomeOr = requireAscendHome(); !ascendHomeOr)
    return ascendHomeOr.takeError();

  auto layoutOr = buildMixDirectCompileLayout(outputDir, kernelName);
  if (!layoutOr)
    return layoutOr.takeError();
  const MixCompileLayout &layout = *layoutOr;

  const std::string analyzeContext = makeStageContext({
      {"kernel", kernelName},
      {"source", kernelSrc},
      {"soc_version", socVersion},
  });
  llvm::SmallString<256> sourcePath(kernelSrc);
  if (auto ec = llvm::sys::fs::make_absolute(sourcePath))
    return llvm::createStringError(
        ec, "[%s] cannot resolve kernel source path: %s (inputs: %s)",
        kStageAnalyzeSource, kernelSrc.str().c_str(), analyzeContext.c_str());
  if (!llvm::sys::fs::exists(sourcePath))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "[%s] kernel source file not found: %s (inputs: %s)",
        kStageAnalyzeSource, sourcePath.c_str(), analyzeContext.c_str());

  auto analyzed = analyzeMixKernel(sourcePath, kernelName, socVersion);
  if (!analyzed) {
    const std::string analysisError = llvm::toString(analyzed.takeError());
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "[%s] inputs: %s: %s",
                                   kStageAnalyzeSource, analyzeContext.c_str(),
                                   analysisError.c_str());
  }

  auto compileOr = executeMixDirectCompilePipeline(
      layout, sourcePath, kernelName, cannMlirPath, npyDir, socVersion,
      *analyzed);
  if (!compileOr)
    return compileOr.takeError();
  return finalizeMixDirectArtifact(layout, sourcePath, *analyzed, *compileOr);
}

} // namespace mlir::runtime
