#include "MixDirectCompileInternal.h"

#include "Runtime/Mix/MixStubTemplate.h"
#include "Runtime/MixCommandBuilder.h"
#include "Runtime/Support/PathUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir::runtime {
namespace {

static constexpr const char *kStageFinalizeHostStub = "finalize host stub";
static constexpr const char *kStageCompileAic = "compile AIC object";
static constexpr const char *kStageCompileAiv = "compile AIV object";
static constexpr const char *kStageMergeAic = "merge AIC object";
static constexpr const char *kStageMergeAiv = "merge AIV object";
static constexpr const char *kStageMergeDevice = "merge device objects";
static constexpr const char *kStageCompileHostStub = "compile host stub";
static constexpr const char *kStageCompileHostBisheng = "compile host bisheng";
static constexpr const char *kStagePack = "pack mix kernel";
static constexpr const char *kStageLinkHostStub = "link host runner library";
static constexpr const char *kStageRecompile = "recompile packed binary";

static std::vector<std::string>
buildFinalMergeCommand(llvm::StringRef aicObj, llvm::StringRef aivObj,
                       llvm::StringRef outputObj) {
  std::vector<std::string> cmd =
      buildLldMergeCommand(aicObj, aivObj, outputObj);
  cmd.insert(cmd.begin() + 1, "--allow-multiple-definition");
  if (cmd.size() > 2 && cmd[2] == "-r")
    cmd.erase(cmd.begin() + 2);
  return cmd;
}

static llvm::Error copyFileOrErr(llvm::StringRef from, llvm::StringRef to) {
  if (auto ec = llvm::sys::fs::copy_file(from, to))
    return llvm::createStringError(ec, "Cannot copy %s -> %s",
                                   from.str().c_str(), to.str().c_str());
  return llvm::Error::success();
}

static llvm::Expected<uint64_t> getFileSizeOrErr(llvm::StringRef path) {
  uint64_t size = 0;
  if (auto ec = llvm::sys::fs::file_size(path, size))
    return llvm::createStringError(ec, "Cannot stat file: %s",
                                   path.str().c_str());
  return size;
}

static uint64_t alignTo4(uint64_t size) { return (size + 3ULL) & ~3ULL; }

static llvm::Error writeRecompileLinkFile(llvm::StringRef rootDir,
                                          llvm::StringRef targetName,
                                          llvm::StringRef linkCommand) {
  llvm::SmallString<256> linkDir(rootDir);
  llvm::sys::path::append(linkDir, "CMakeFiles", targetName.str() + ".dir");
  if (auto err = ensureDirectory(linkDir))
    return err;
  llvm::SmallString<256> linkPath(linkDir);
  llvm::sys::path::append(linkPath, "link.txt");
  return writeTextFile(linkPath, linkCommand.str() + "\n");
}

} // namespace

llvm::Expected<MixDirectBuildOutputs>
executeMixDirectBinaryBuild(const MixDirectCompileContract &contract,
                            llvm::StringRef sourcePath,
                            llvm::StringRef requestedKernelName,
                            llvm::StringRef socVersion) {
  const MixCompileLayout &layout = contract.layout;
  MixDirectBuildOutputs outputs;
  outputs.runtimeKernelName = contract.runtimeKernelName;
  outputs.generatedSourcePath = contract.generatedSourcePath;
  outputs.hostSourcePath = sourcePath.str();
  outputs.hostStubSourcePath = contract.preprocess.hostStubPath;
  outputs.hostStubIncludeDir = contract.preprocess.includeDir;
  outputs.preprocessIncludeDir = contract.preprocess.includeDir;
  outputs.preprocessCompileCommandsPath = contract.preprocess.compileCommandsPath;
  outputs.preprocessCommand = contract.preprocess.preprocessCommand;
  outputs.preprocessGeneratedDir = contract.preprocess.generatedDir;

  const std::string runnerLauncherCopyPath =
      joinPath(layout.outIncludeDir,
               "aclrtlaunch_" + outputs.runtimeKernelName + ".h");
  const bool needsManualStubTemplate =
      contract.preprocess.launcherHeaderPath.empty() ||
      contract.synthesizedAicFromAiv;
  std::string launcherHeaderPath = contract.preprocess.launcherHeaderPath;
  if (needsManualStubTemplate) {
    outputs.hostStubSourcePath = joinPath(layout.stubDir, "host_stub.cpp");
    outputs.hostStubIncludeDir = layout.outIncludeDir;
    launcherHeaderPath = runnerLauncherCopyPath;
  }

  const std::vector<std::string> aicCmd =
      buildPreprocessedDeviceCompileCommand(outputs.generatedSourcePath,
                                            layout.aicObj, MixCoreType::AIC,
                                            contract.aicDefinitions);
  const std::vector<std::string> aivCmd =
      buildPreprocessedDeviceCompileCommand(outputs.generatedSourcePath,
                                            layout.aivObj, MixCoreType::AIV,
                                            contract.aivDefinitions);
  const std::vector<std::string> aicRelocCmd =
      buildLldRelocCommand(layout.aicObj, layout.aicRelocObj);
  const std::vector<std::string> aivRelocCmd =
      buildLldRelocCommand(layout.aivObj, layout.aivRelocObj);
  const std::vector<std::string> mergeCmd = buildFinalMergeCommand(
      layout.aicRelocObj, layout.aivRelocObj, layout.mergeDeviceObj);
  const std::vector<std::string> hostCompileCmd =
      buildHostStubCompileCommand(outputs.hostStubSourcePath,
                                  layout.hostStubObjectPath,
                                  outputs.hostStubIncludeDir);
  const std::vector<std::string> packCmd =
      buildPackCommand(layout.hostStubObjectPath, layout.mergeDir);
  const std::vector<std::string> hostLinkCmd =
      buildHostSharedLinkCommand(layout.hostStubObjectPath, layout.kernelSoPath,
                                 socVersion,
                                 findAscendDeviceLibDir(findAscendHome()));

  const std::string aicCompileContext = makeStageContext({
      {"kernel", requestedKernelName},
      {"source", outputs.generatedSourcePath},
      {"output", layout.aicObj},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string aivCompileContext = makeStageContext({
      {"kernel", requestedKernelName},
      {"source", outputs.generatedSourcePath},
      {"output", layout.aivObj},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string aicMergeContext = makeStageContext({
      {"input", layout.aicObj},
      {"output", layout.aicRelocObj},
      {"kernel", requestedKernelName},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string aivMergeContext = makeStageContext({
      {"input", layout.aivObj},
      {"output", layout.aivRelocObj},
      {"kernel", requestedKernelName},
      {"generated_dir", outputs.preprocessGeneratedDir},
  });
  const std::string mergeContext = makeStageContext({
      {"aic_input", layout.aicRelocObj},
      {"aiv_input", layout.aivRelocObj},
      {"output", layout.mergeDeviceObj},
      {"kernel", requestedKernelName},
  });
  const std::string finalizeContext = makeStageContext({
      {"generated_dir", outputs.preprocessGeneratedDir},
      {"merge_dir", layout.mergeDir},
      {"soc_version", socVersion},
      {"target", "ascendc_kernels_sim"},
  });
  const std::string hostCompileContext = makeStageContext({
      {"source", outputs.hostStubSourcePath},
      {"output", layout.hostStubObjectPath},
      {"include_dir", outputs.hostStubIncludeDir},
      {"kernel", requestedKernelName},
  });
  const std::string packContext = makeStageContext({
      {"input", layout.hostStubObjectPath},
      {"add_dir", layout.mergeDir},
      {"kernel", requestedKernelName},
  });
  const std::string hostLinkContext = makeStageContext({
      {"input", layout.hostStubObjectPath},
      {"output", layout.kernelSoPath},
      {"soc_version", socVersion},
      {"kernel", requestedKernelName},
  });
  const std::string tripleChevronHeaderPath =
      joinPath(outputs.preprocessIncludeDir.empty() ? outputs.hostStubIncludeDir
                                                    : outputs.preprocessIncludeDir,
               "aclrtlaunch_triple_chevrons_func.h");
  outputs.hostBishengObjectPath =
      joinPath(layout.hostObjectsDir,
               llvm::sys::path::filename(outputs.hostSourcePath).str() + ".o");
  outputs.hostObjectDir = layout.hostDir;
  const std::vector<std::string> hostBishengCmd =
      buildHostBishengCommand(outputs.hostSourcePath,
                              outputs.hostBishengObjectPath,
                              tripleChevronHeaderPath);
  const std::vector<std::string> recompileCmd =
      buildRecompileBinaryCommand(layout.outputRoot, "ascendc_kernels_sim",
                                  layout.hostDir);

  {
    MixDirectStageTimer timer("device_compile_parallel", outputs.timings);
    if (auto err = runProcessesInParallel({
            {aicCmd, kStageCompileAic, aicCompileContext},
            {aivCmd, kStageCompileAiv, aivCompileContext},
        }))
      return std::move(err);
  }
  if (auto err =
          ensureFileExists(layout.aicObj, kStageCompileAic, aicCompileContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.aivObj, kStageCompileAiv, aivCompileContext))
    return std::move(err);
  {
    MixDirectStageTimer timer("device_reloc_parallel", outputs.timings);
    if (auto err = runProcessesInParallel({
            {aicRelocCmd, kStageMergeAic, aicMergeContext},
            {aivRelocCmd, kStageMergeAiv, aivMergeContext},
        }))
      return std::move(err);
  }
  if (auto err =
          ensureFileExists(layout.aicRelocObj, kStageMergeAic, aicMergeContext))
    return std::move(err);
  if (auto err =
          ensureFileExists(layout.aivRelocObj, kStageMergeAiv, aivMergeContext))
    return std::move(err);
  {
    MixDirectStageTimer timer("merge_device_objects", outputs.timings);
    if (auto err = runProcess(mergeCmd, kStageMergeDevice, mergeContext))
      return std::move(err);
  }
  if (auto err = ensureFileExists(layout.mergeDeviceObj, kStageMergeDevice,
                                  mergeContext))
    return std::move(err);
  uint64_t mixFileLen = 0;
  {
    MixDirectStageTimer timer("finalize_device_object", outputs.timings);
    if (auto err = copyFileOrErr(layout.mergeDeviceObj, layout.mergedDeviceObj))
      return std::move(err);
    auto mixLen = getFileSizeOrErr(layout.mergedDeviceObj);
    if (!mixLen)
      return mixLen.takeError();
    mixFileLen = *mixLen;
    if (auto err = writeTextFile(layout.mixFlagPath, ""))
      return std::move(err);
  }

  if (needsManualStubTemplate) {
    MixDirectStageTimer timer("write_manual_host_stub", outputs.timings);
    MixStubTemplateArgs stubArgs;
    stubArgs.kernelName = outputs.runtimeKernelName;
    stubArgs.targetName = "ascendc_kernels_sim";
    stubArgs.socVersion = socVersion.str();
    stubArgs.launcherSymbol = "aclrtlaunch_" + outputs.runtimeKernelName;
    stubArgs.launcherHeaderPath = runnerLauncherCopyPath;
    stubArgs.hostStubSourcePath = outputs.hostStubSourcePath;
    stubArgs.mixLen = alignTo4(mixFileLen);
    stubArgs.mixFileLen = mixFileLen;
    stubArgs.aivOnly = false;
    if (auto err = writeMixStubTemplate(stubArgs))
      return std::move(err);
    launcherHeaderPath = runnerLauncherCopyPath;
  } else {
    const std::string lowerSocVersion = llvm::StringRef(socVersion).lower();
    const std::vector<std::string> finalizeHostStubCmd =
        buildUpdateHostStubCommand(outputs.preprocessGeneratedDir,
                                   layout.mergeDir, lowerSocVersion,
                                   "ascendc_kernels_sim");
    {
      MixDirectStageTimer timer("finalize_host_stub", outputs.timings);
      if (auto err = runProcess(finalizeHostStubCmd, kStageFinalizeHostStub,
                                finalizeContext))
        return std::move(err);
      if (auto err = copyFileOrErr(launcherHeaderPath, runnerLauncherCopyPath))
        return std::move(err);
    }
  }

  {
    MixDirectStageTimer timer("compile_host_stub", outputs.timings);
    if (auto err = runProcess(hostCompileCmd, kStageCompileHostStub,
                              hostCompileContext))
      return std::move(err);
  }
  if (auto err = ensureFileExists(layout.hostStubObjectPath,
                                  kStageCompileHostStub, hostCompileContext))
    return std::move(err);
  const std::string hostBishengContext = makeStageContext({
      {"source", outputs.hostSourcePath},
      {"output", outputs.hostBishengObjectPath},
      {"triple_chevron_header", tripleChevronHeaderPath},
      {"kernel", requestedKernelName},
  });
  if (auto err = ensureFileExists(tripleChevronHeaderPath,
                                  kStageCompileHostBisheng,
                                  hostBishengContext))
    return std::move(err);
  {
    MixDirectStageTimer timer("compile_host_bisheng", outputs.timings);
    if (auto err = runProcess(hostBishengCmd, kStageCompileHostBisheng,
                              hostBishengContext))
      return std::move(err);
  }
  if (auto err = ensureFileExists(outputs.hostBishengObjectPath,
                                  kStageCompileHostBisheng,
                                  hostBishengContext))
    return std::move(err);
  {
    MixDirectStageTimer timer("pack_mix_kernel", outputs.timings);
    if (auto err = runProcess(packCmd, kStagePack, packContext))
      return std::move(err);
  }
  if (auto err =
          ensureFileExists(layout.hostStubObjectPath, kStagePack, packContext))
    return std::move(err);
  {
    MixDirectStageTimer timer("link_host_runner_library", outputs.timings);
    if (auto err = runProcess(hostLinkCmd, kStageLinkHostStub, hostLinkContext))
      return std::move(err);
  }
  if (auto err = ensureFileExists(layout.kernelSoPath, kStageLinkHostStub,
                                  hostLinkContext))
    return std::move(err);

  {
    MixDirectStageTimer timer("prepare_recompile_link_file", outputs.timings);
    const std::string recompileHostStubObjectPath =
        joinPath(layout.stubDir, "host_stub.cpp.o");
    if (auto err = copyFileOrErr(layout.hostStubObjectPath,
                                 recompileHostStubObjectPath))
      return std::move(err);
    std::vector<std::string> recompileLinkArgs = hostLinkCmd;
    for (std::string &arg : recompileLinkArgs) {
      if (arg == layout.hostStubObjectPath)
        arg = recompileHostStubObjectPath;
    }
    const std::string recompileLinkCmd =
        renderCommandForCompileCommands(recompileLinkArgs);
    if (auto err = writeRecompileLinkFile(layout.outputRoot,
                                          "ascendc_kernels_sim",
                                          recompileLinkCmd))
      return std::move(err);
  }
  const std::string recompileContext = makeStageContext({
      {"root_dir", layout.outputRoot},
      {"target_name", "ascendc_kernels_sim"},
      {"add_dir", layout.hostDir},
      {"kernel", requestedKernelName},
  });
  {
    MixDirectStageTimer timer("recompile_packed_binary", outputs.timings);
    if (auto err = runProcess(recompileCmd, kStageRecompile, recompileContext))
      return std::move(err);
  }
  const std::string recompiledKernelSoPath =
      joinPath(layout.outDir, "lib" + outputs.runtimeKernelName + "_packed.so");
  if (auto err = ensureFileExists(recompiledKernelSoPath, kStageRecompile,
                                  recompileContext))
    return std::move(err);

  outputs.aicCompileCommand = renderCommandForDebug(aicCmd);
  outputs.aivCompileCommand = renderCommandForDebug(aivCmd);
  outputs.aicRelocCommand = renderCommandForDebug(aicRelocCmd);
  outputs.aivRelocCommand = renderCommandForDebug(aivRelocCmd);
  outputs.mergeCommand = renderCommandForDebug(mergeCmd);
  outputs.hostCompileCommand = renderCommandForDebug(hostCompileCmd);
  outputs.hostBishengCommand = renderCommandForDebug(hostBishengCmd);
  outputs.packCommand = renderCommandForDebug(packCmd);
  outputs.hostLinkCommand = renderCommandForDebug(hostLinkCmd);
  outputs.recompileCommand = renderCommandForDebug(recompileCmd);
  return outputs;
}

} // namespace mlir::runtime
