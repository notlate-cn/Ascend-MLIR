#pragma once

#include "Runtime/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::runtime {

struct MixAbiTensorDesc {
  std::string name;
  std::string runtimeFile;
  std::string goldenFile;
  DType dtype = DType::F16;
  std::vector<int64_t> shape;
};

struct MixAbiMetadata {
  std::string logicalKernelName;
  std::string runtimeKernelName;
  std::vector<MixAbiTensorDesc> inputs;
  std::vector<MixAbiTensorDesc> outputs;
  std::optional<size_t> workspaceArgIndex;
  std::optional<size_t> tilingArgIndex;
  size_t workspaceBytes = 0;
  uint32_t blockDim = 1;
  std::string workspaceMode;
  std::string tilingMode;
  std::string tilingSource;
  std::string launcherSymbol;
  std::string aicEntry;
  std::string aivEntry;
};

std::string buildCanonicalInputFileName(llvm::StringRef kernelName,
                                        llvm::StringRef tensorName);
std::string buildCanonicalOutputFileName(llvm::StringRef kernelName,
                                         llvm::StringRef tensorName);
std::string buildCanonicalGoldenFileName(llvm::StringRef kernelName,
                                         llvm::StringRef tensorName);
std::string normalizeMixKernelName(llvm::StringRef kernelName);

llvm::Expected<std::string>
serializeMixAbiManifest(const MixAbiMetadata &abi);

llvm::Expected<MixAbiMetadata>
parseMixAbiManifest(const std::map<std::string, std::string> &manifest);

} // namespace mlir::runtime
