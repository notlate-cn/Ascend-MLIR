#pragma once

#include "Runtime/Support/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::runtime {

struct MixCompileMetadataEntries {
  std::string aic;
  std::string aiv;
};

struct MixCompileMetadataGenerated {
  std::string sourcePath;
};

struct MixCompileMetadataDeviceCompile {
  std::string aicArch;
  std::string aivArch;
  std::vector<std::string> aicDefinitions;
  std::vector<std::string> aivDefinitions;
};

struct MixCompileMetadataArtifacts {
  std::string deviceObjectPath;
  std::string packedSharedObjectPath;
  std::string tilingFilePath;
  std::string launchInfoFilePath;
};

struct MixCompileMetadataTensorDesc {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string runtimeFile;
};

struct MixCompileMetadataAbi {
  std::string workspaceMode;
  uint64_t workspaceBytes = 0;
  std::string tilingMode;
  std::string tilingSource;
  std::vector<MixCompileMetadataTensorDesc> inputs;
  std::vector<MixCompileMetadataTensorDesc> outputs;
};

struct MixCompileMetadataHostLaunch {
  std::string mode;
  std::string helperKind;
  std::string helperInputsJson;
};

struct MixCompileMetadata {
  uint64_t schemaVersion = 0;
  std::string kernelKind;
  std::string kernelName;
  std::string runtimeKernelName;
  std::string socVersion;
  std::string mixKernelType;
  std::string launcherSymbol;
  MixCompileMetadataEntries entries;
  MixCompileMetadataGenerated generated;
  MixCompileMetadataDeviceCompile deviceCompile;
  MixCompileMetadataArtifacts artifacts;
  MixCompileMetadataAbi abi;
  MixCompileMetadataHostLaunch hostLaunch;
};

llvm::Expected<MixCompileMetadata>
parseMixCompileMetadataJson(llvm::StringRef jsonText);

llvm::Expected<std::string>
serializeMixCompileMetadataJson(const MixCompileMetadata &metadata);

} // namespace mlir::runtime
