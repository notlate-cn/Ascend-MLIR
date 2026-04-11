#pragma once

#include "Runtime/ArtifactCompiler.h"
#include "Runtime/RunManifest.h"

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

ArtifactCompileRequest
buildCompatCompileRequest(const CompatCompileOptions &options);

struct CompatValidateOptions {
  std::string artifactRoot;
  std::vector<std::string> inputPaths;
  std::string expectedOutputPath;
  std::string actualOutputPath;
  std::string tilingSchemaPath;
  std::string tilingParams;
  std::string tilingBinaryPath;
  int blockDim = 1;
  double atol = 1.0;
  double rtol = 1e-2;
};

RunManifestSpec
buildCompatSingleTaskRunManifest(const CompatValidateOptions &options);

} // namespace mlir::runtime
