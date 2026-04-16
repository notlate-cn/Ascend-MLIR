#pragma once

#include "Runtime/Support/Types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::runtime {

struct MixTilingTensorDesc {
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
};

struct MixTilingRequest {
  std::string kernelName;
  std::string socVersion;
  std::vector<MixTilingTensorDesc> inputs;
  std::vector<MixTilingTensorDesc> outputs;
};

struct MixTilingResult {
  std::vector<uint8_t> tilingData;
  uint32_t blockDim = 0;
  std::string strategyName;
};

llvm::StringRef getDefaultMixTilingBackendName();

llvm::Expected<MixTilingResult>
generateMixTilingInProcess(const MixTilingRequest &request);

llvm::Error writeMixTilingArtifacts(const MixTilingResult &result,
                                    llvm::StringRef tilingOutputPath,
                                    llvm::StringRef launchInfoOutputPath);

} // namespace mlir::runtime
