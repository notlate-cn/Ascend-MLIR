#pragma once

#include "Runtime/Mix/MixSourceAnalyzer.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
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

struct MixLegacyCompileContract {
  MixPreprocessOutputs preprocess;
  MixCompileLayout layout;
  std::string generatedSourceName;
  std::string generatedSourcePath;
  std::string runtimeKernelName;
  std::vector<std::string> aicDefinitions;
  std::vector<std::string> aivDefinitions;
  bool synthesizedAicFromAiv = false;
};

llvm::Expected<MixGeneratedConfig>
parseMixGeneratedConfig(llvm::StringRef path);

llvm::Expected<MixPreprocessOutputs>
runLegacyMixPreprocessStage(llvm::StringRef workDir, llvm::StringRef sourcePath,
                            llvm::StringRef kernelName,
                            llvm::StringRef socVersion,
                            llvm::StringRef aivProbeObject,
                            llvm::StringRef aicProbeObject);

llvm::Expected<MixLegacyCompileContract> loadLegacyMixCompileContract(
    const MixCompileLayout &layout, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    const MixAnalyzedKernel &analyzed);

llvm::Expected<MixCompileLayout>
buildLegacyMixCompileLayout(llvm::StringRef outputDir, llvm::StringRef kernelName);

} // namespace mlir::runtime
