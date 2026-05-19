#pragma once

#include "Runtime/Mix/MixAbi.h"
#include "Runtime/Mix/MixArtifact.h"
#include "Runtime/Mix/MixSourceAnalyzer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::runtime {

struct MixPreprocessOutputs {
  std::string preprocessedSourcePath;
  std::string compileCommandsPath;
  std::string preprocessCommand;
  std::string generatedDir;
  std::string includeDir;
  std::string hostStubPath;
  std::string launcherHeaderPath;
  std::string actualLauncherKernelName;
  std::string aicConfigPath;
  std::string aivConfigPath;
};

enum class MixDirectContractMode {
  DirectSource,
};

struct MixCompileLayout {
  std::string outputRoot;
  std::string workDir;
  std::string objectDir;
  std::string outDir;
  std::string outBinDir;
  std::string outIncludeDir;
  std::string mergeDir;
  std::string launcherDir;
  std::string stubDir;
  std::string hostDir;
  std::string hostObjectsDir;
  std::string aicMergeDir;
  std::string aivMergeDir;
  std::string aicObj;
  std::string aivObj;
  std::string aicRelocObj;
  std::string aivRelocObj;
  std::string mergedDeviceObj;
  std::string manifestPath;
  std::string metadataPath;
  std::string timingPath;
  std::string analysisPath;
  std::string mergeDeviceObj;
  std::string hostStubObjectPath;
  std::string kernelSoPath;
  std::string mixFlagPath;
  std::string tilingArtifactPath;
  std::string launchInfoPath;
};

struct MixDirectKernelArg {
  std::string type;
  std::string name;
  bool passTilingByValue = false;
};

struct MixDirectCompileContract {
  MixDirectContractMode mode = MixDirectContractMode::DirectSource;
  MixPreprocessOutputs preprocess;
  MixCompileLayout layout;
  std::string generatedSourceName;
  std::string generatedSourcePath;
  std::string runtimeKernelName;
  std::vector<MixDirectKernelArg> kernelArgs;
  std::optional<size_t> workspaceArgIndex;
  std::optional<size_t> tilingArgIndex;
  std::vector<std::string> aicDefinitions;
  std::vector<std::string> aivDefinitions;
  bool synthesizedAicFromAiv = false;
};

struct MixDirectTimingEntry {
  std::string name;
  uint64_t elapsedUs = 0;
};

struct MixDirectBuildOutputs {
  std::string runtimeKernelName;
  std::string generatedSourcePath;
  std::string hostSourcePath;
  std::string hostStubSourcePath;
  std::string hostStubIncludeDir;
  std::string preprocessIncludeDir;
  std::string preprocessCompileCommandsPath;
  std::string preprocessCommand;
  std::string preprocessGeneratedDir;
  std::string hostObjectDir;
  std::string aicCompileCommand;
  std::string aivCompileCommand;
  std::string aicRelocCommand;
  std::string aivRelocCommand;
  std::string mergeCommand;
  std::string hostCompileCommand;
  std::string packCommand;
  std::string hostLinkCommand;
  std::vector<MixDirectTimingEntry> timings;
};

struct MixDirectTilingOutputs {
  uint32_t blockDim = 0;
  std::string tilingArtifactPath;
  std::string launchInfoPath;
  std::string backendKind;
  std::string strategyName;
  std::string debugNote;
  std::string runnerCompileCommand;
  std::string tilingEmitCommand;
  std::vector<MixDirectTimingEntry> timings;
};

struct MixDirectCompileOutputs {
  MixDirectCompileContract contract;
  MixDirectBuildOutputs build;
  MixAbiMetadata abi;
  MixDirectTilingOutputs tiling;
  std::string metadataPath;
  std::vector<MixDirectTimingEntry> timings;
};

struct MixDirectDebugManifestInputs {
  const MixAnalyzedKernel *analyzed = nullptr;
  const MixAbiMetadata *abi = nullptr;
  std::string runtimeKernelName;
  std::string sourcePath;
  std::string hostSourcePath;
  std::string preprocessCompileCommandsPath;
  std::string preprocessCommand;
  std::string preprocessGeneratedDir;
  std::string generatedSourcePath;
  std::string aicDefinitions;
  std::string aivDefinitions;
  std::string workDir;
  std::string objectDir;
  std::string outDir;
  std::string mergeDir;
  std::string launcherHeaderDir;
  std::string hostStubSourcePath;
  std::string hostStubObjectPath;
  std::string kernelSoPath;
  std::string mixFlagPath;
  std::string aicObj;
  std::string aivObj;
  std::string aicRelocObj;
  std::string aivRelocObj;
  std::string mergedDeviceObj;
  std::string aicCompileCmd;
  std::string aivCompileCmd;
  std::string aicRelocCmd;
  std::string aivRelocCmd;
  std::string mergeCmd;
  std::string hostCompileCmd;
  std::string hostObjectDir;
  std::string packCmd;
  std::string linkCmd;
  std::string tilingBackendKind;
  std::string tilingStrategyName;
  std::string tilingDebugNote;
  std::string runnerCompileCmd;
  std::string metadataPath;
  std::string manifestPath;
};

struct MixDirectProcessCommand {
  std::vector<std::string> args;
  std::string stage;
  std::string context;
};

class MixDirectStageTimer {
public:
  MixDirectStageTimer(llvm::StringRef name,
                      std::vector<MixDirectTimingEntry> &entries);
  ~MixDirectStageTimer();

  MixDirectStageTimer(const MixDirectStageTimer &) = delete;
  MixDirectStageTimer &operator=(const MixDirectStageTimer &) = delete;

private:
  std::string name;
  std::vector<MixDirectTimingEntry> &entries;
  std::chrono::steady_clock::time_point start;
};

llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content);

llvm::Expected<std::string> readTextFileOrErr(llvm::StringRef path);

llvm::Error ensureDirectory(llvm::StringRef path);

llvm::Error ensureFileExists(llvm::StringRef path, llvm::StringRef stage,
                             llvm::StringRef context = {});

std::string joinPath(llvm::StringRef base, llvm::StringRef leaf);

std::string makeStageContext(
    std::initializer_list<std::pair<llvm::StringRef, llvm::StringRef>> fields);

llvm::Error runProcess(const std::vector<std::string> &args,
                       llvm::StringRef stage, llvm::StringRef context = {});

llvm::Error
runProcessesInParallel(llvm::ArrayRef<MixDirectProcessCommand> commands);

llvm::Expected<std::string>
serializeMixDirectTimingJson(llvm::ArrayRef<MixDirectTimingEntry> entries);

llvm::Error
writeMixDirectTimingFile(llvm::StringRef path,
                         llvm::ArrayRef<MixDirectTimingEntry> entries);

llvm::Expected<MixDirectCompileContract> loadMixDirectCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    const MixAnalyzedKernel &analyzed);

llvm::Expected<MixDirectCompileContract> buildMixDirectSourceCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, const MixAnalyzedKernel &analyzed);

llvm::Expected<std::pair<std::string, std::string>>
writeMixDirectManualHostStub(const MixDirectCompileContract &contract,
                             llvm::StringRef socVersion, uint64_t mixFileLen,
                             bool aivOnly);

llvm::Expected<MixCompileLayout>
buildMixDirectCompileLayout(llvm::StringRef outputDir,
                            llvm::StringRef kernelName);

llvm::Expected<MixDirectBuildOutputs>
executeMixDirectBinaryBuild(const MixDirectCompileContract &contract,
                            llvm::StringRef sourcePath,
                            llvm::StringRef requestedKernelName,
                            llvm::StringRef socVersion);

llvm::Expected<MixAbiMetadata>
loadMixDirectRuntimeAbi(llvm::StringRef cannMlirPath, llvm::StringRef npyDir,
                        llvm::StringRef runtimeKernelName);

llvm::Expected<std::string>
writeMixDirectCompileMetadataFile(llvm::StringRef metadataPath,
                                  llvm::StringRef runtimeKernelName,
                                  llvm::StringRef socVersion,
                                  llvm::StringRef mixKernelType,
                                  llvm::StringRef generatedSourcePath,
                                  llvm::ArrayRef<std::string> aicDefinitions,
                                  llvm::ArrayRef<std::string> aivDefinitions,
                                  llvm::StringRef deviceObjectPath,
                                  llvm::StringRef packedSharedObjectPath,
                                  const MixDirectTilingOutputs &tiling,
                                  const MixAbiMetadata &abi);

llvm::Error
writeMixDirectDebugManifest(const MixDirectDebugManifestInputs &inputs);

llvm::Expected<MixDirectTilingOutputs>
executeMixDirectTilingStage(const MixCompileLayout &layout,
                            llvm::StringRef runtimeKernelName,
                            llvm::StringRef socVersion,
                            const MixAbiMetadata &abi);

llvm::Expected<MixDirectCompileOutputs>
executeMixDirectCompilePipeline(const MixCompileLayout &layout,
                                llvm::StringRef sourcePath,
                                llvm::StringRef requestedKernelName,
                                llvm::StringRef cannMlirPath,
                                llvm::StringRef npyDir,
                                llvm::StringRef socVersion,
                                const MixAnalyzedKernel &analyzed);

llvm::Expected<MixArtifact>
finalizeMixDirectArtifact(const MixCompileLayout &layout,
                          llvm::StringRef sourcePath,
                          const MixAnalyzedKernel &analyzed,
                          const MixDirectCompileOutputs &compile);

} // namespace mlir::runtime
