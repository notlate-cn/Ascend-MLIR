#pragma once

#include "Runtime/Support/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir::runtime {

enum class MixCoreType {
  AIC,
  AIV,
};

std::vector<std::string>
buildPreprocessedDeviceCompileCommand(llvm::StringRef src,
                                      llvm::StringRef obj,
                                      MixCoreType coreType,
                                      llvm::ArrayRef<std::string> defs = {});

std::vector<std::string> buildLldRelocCommand(llvm::StringRef inputObj,
                                              llvm::StringRef outputObj);

std::vector<std::string> buildLldMergeCommand(llvm::StringRef aicObj,
                                              llvm::StringRef aivObj,
                                              llvm::StringRef outputObj);

std::vector<std::string> buildHostStubCompileCommand(llvm::StringRef source,
                                                     llvm::StringRef object,
                                                     llvm::StringRef headerDir);

std::vector<std::string> buildPackCommand(llvm::StringRef hostStubObject,
                                          llvm::StringRef addDir);

std::vector<std::string> buildHostSharedLinkCommand(llvm::StringRef hostStubObject,
                                                    llvm::StringRef outputSo,
                                                    llvm::StringRef socVersion,
                                                    llvm::StringRef deviceLibDir);

std::vector<std::string>
buildMixTilingHelperCommand(llvm::StringRef kernelName,
                            llvm::StringRef socVersion,
                            llvm::ArrayRef<int64_t> inputAShape,
                            DType inputADType,
                            llvm::ArrayRef<int64_t> inputBShape,
                            DType inputBDType,
                            llvm::ArrayRef<int64_t> outputShape,
                            DType outputDType,
                            const std::optional<DType> &biasDType,
                            llvm::StringRef tilingOutputPath,
                            llvm::StringRef launchInfoOutputPath);

std::string renderCommandForDebug(llvm::ArrayRef<std::string> args);

} // namespace mlir::runtime
