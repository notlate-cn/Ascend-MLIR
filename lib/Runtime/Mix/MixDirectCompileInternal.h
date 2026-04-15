#pragma once

#include "Runtime/Mix/MixAbi.h"
#include "Runtime/Mix/MixArtifact.h"
#include "Runtime/Mix/MixSourceAnalyzer.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace mlir::runtime {

struct MixGeneratedConfig {
  std::vector<std::string> mixSources;
  llvm::StringMap<std::vector<std::string>> definitionsBySource;
};

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
  std::string analysisPath;
  std::string mergeDeviceObj;
  std::string hostStubObjectPath;
  std::string kernelSoPath;
  std::string mixFlagPath;
  std::string runnerMainPath;
  std::string runnerTilingPath;
  std::string runnerDataUtilsPath;
  std::string runnerBinaryPath;
  std::string tilingArtifactPath;
  std::string launchInfoPath;
  std::string preprocessProbeDir;
  std::string aicProbeObject;
  std::string aivProbeObject;
};

struct MixDirectCompileContract {
  MixPreprocessOutputs preprocess;
  MixCompileLayout layout;
  std::string generatedSourceName;
  std::string generatedSourcePath;
  std::string runtimeKernelName;
  std::vector<std::string> aicDefinitions;
  std::vector<std::string> aivDefinitions;
  bool synthesizedAicFromAiv = false;
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
  std::string hostBishengObjectPath;
  std::string hostObjectDir;
  std::string aicCompileCommand;
  std::string aivCompileCommand;
  std::string aicRelocCommand;
  std::string aivRelocCommand;
  std::string mergeCommand;
  std::string hostCompileCommand;
  std::string hostBishengCommand;
  std::string packCommand;
  std::string hostLinkCommand;
  std::string recompileCommand;
};

struct MixDirectTilingOutputs {
  bool usedLegacyRunner = false;
  uint32_t blockDim = 0;
  std::string tilingArtifactPath;
  std::string launchInfoPath;
  std::string runnerSourcePath;
  std::string runnerBinaryPath;
  std::string runnerCompileCommand;
  std::string tilingEmitCommand;
};

struct MixDirectCompileOutputs {
  MixDirectCompileContract contract;
  MixDirectBuildOutputs build;
  MixAbiMetadata abi;
  MixDirectTilingOutputs tiling;
  std::string metadataPath;
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
  std::string runnerSourcePath;
  std::string runnerBinaryPath;
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
  std::string hostBishengObjectPath;
  std::string hostBishengCmd;
  std::string hostObjectDir;
  std::string packCmd;
  std::string linkCmd;
  std::string recompileCmd;
  std::string runnerCompileCmd;
  std::string metadataPath;
  std::string manifestPath;
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

llvm::Expected<MixGeneratedConfig>
parseMixGeneratedConfig(llvm::StringRef path);

llvm::Expected<MixPreprocessOutputs>
runMixDirectPreprocessStage(llvm::StringRef workDir, llvm::StringRef sourcePath,
                            llvm::StringRef kernelName,
                            llvm::StringRef socVersion,
                            llvm::StringRef aivProbeObject,
                            llvm::StringRef aicProbeObject);

llvm::Expected<MixDirectCompileContract> loadMixDirectCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    const MixAnalyzedKernel &analyzed);

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
                                  llvm::StringRef tilingFilePath,
                                  llvm::StringRef launchInfoFilePath,
                                  const MixAbiMetadata &abi,
                                  bool useLegacyRunner);

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
