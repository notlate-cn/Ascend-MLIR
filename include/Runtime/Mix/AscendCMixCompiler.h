// include/Runtime/AscendCMixCompiler.h
#pragma once
#include "Runtime/Mix/MixArtifact.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

struct AscendCMixCompileConfig {
  std::string kernel_src;
  std::string kernel_name;
  std::string soc_version;
  std::string output_dir;
  std::string ascend_cmake_dir;
};

class AscendCMixCompiler {
public:
  llvm::Expected<MixArtifact> Compile(const AscendCMixCompileConfig& cfg);
};

} // namespace mlir::runtime
