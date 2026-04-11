#pragma once

#include "Runtime/ArtifactCompiler.h"
#include "Runtime/RunManifest.h"

#include <optional>

namespace mlir::runtime {

struct CompatCompileOptions {
  std::string kernelSourcePath;
  std::string outputRoot;
  std::string requestedKernelName;
  std::string socVersion;
  std::string arch = "dav-c220-vec";
  std::string kernelType = "vec";
  bool verbose = false;
};

llvm::Expected<ArtifactCompileRequest>
buildCompatCompileRequest(const CompatCompileOptions &options);

struct CompatValidateOptions {
  std::string artifactRoot;
  std::vector<std::string> inputPaths;
  std::string expectedOutputPath;
  std::string actualOutputPath;
  std::string tilingSchemaPath;
  std::string tilingParams;
  std::string tilingBinaryPath;
  std::optional<std::vector<int64_t>> actualOutputShape;
  std::optional<DType> actualOutputDType;
  int blockDim = 1;
  double atol = 1.0;
  double rtol = 1e-2;
};

llvm::Expected<RunManifestSpec>
buildCompatSingleTaskRunManifest(const CompatValidateOptions &options);

} // namespace mlir::runtime
