#pragma once

#include "RuntimeMix/MixSourceAnalyzer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir::runtime {

enum class MixCoreType {
  AIC,
  AIV,
};

std::vector<std::string> buildBishengCommand(const MixAnalyzedKernel &info,
                                             llvm::StringRef src,
                                             llvm::StringRef obj,
                                             MixCoreType coreType);

std::vector<std::string> buildLldRelocCommand(llvm::StringRef inputObj,
                                              llvm::StringRef outputObj);

std::vector<std::string> buildLldMergeCommand(llvm::StringRef aicObj,
                                              llvm::StringRef aivObj,
                                              llvm::StringRef outputObj);

std::vector<std::string> buildDeviceMergeCommand(llvm::StringRef inputObj,
                                                 llvm::StringRef outputDir,
                                                 llvm::StringRef outputName,
                                                 llvm::StringRef flagPath,
                                                 llvm::StringRef buildType);

std::vector<std::string> buildMixFinalMergeCommand(llvm::StringRef aicDir,
                                                   llvm::StringRef aivDir,
                                                   llvm::StringRef outputDir,
                                                   llvm::StringRef buildType);

std::vector<std::string>
buildPreprocessCommand(llvm::StringRef src, llvm::StringRef outputPath);

std::vector<std::string>
buildExtractHostStubCommand(llvm::StringRef preprocessedPath,
                            llvm::StringRef dstDir,
                            llvm::StringRef headerDir,
                            llvm::ArrayRef<std::string> aivObjects,
                            llvm::ArrayRef<std::string> aicObjects,
                            llvm::StringRef compileCommandsPath,
                            llvm::StringRef buildMode,
                            llvm::StringRef runMode);

std::vector<std::string>
buildUpdateHostStubCommand(llvm::StringRef codeDir, llvm::StringRef objDir,
                           llvm::StringRef lowerSocVersion,
                           llvm::StringRef targetName);

std::vector<std::string> buildHostStubCompileCommand(llvm::StringRef source,
                                                     llvm::StringRef object,
                                                     llvm::StringRef headerDir);

std::vector<std::string> buildPackCommand(llvm::StringRef hostStubObject,
                                          llvm::StringRef addDir);

std::vector<std::string> buildHostSharedLinkCommand(llvm::StringRef hostStubObject,
                                                    llvm::StringRef outputSo,
                                                    llvm::StringRef socVersion);

} // namespace mlir::runtime
