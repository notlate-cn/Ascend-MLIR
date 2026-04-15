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

struct MixLegacyCompileContract {
  MixPreprocessOutputs preprocess;
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
    llvm::StringRef workDir, llvm::StringRef sourcePath,
    llvm::StringRef kernelName, llvm::StringRef socVersion,
    llvm::StringRef aivProbeObject, llvm::StringRef aicProbeObject,
    const MixAnalyzedKernel &analyzed);

} // namespace mlir::runtime
