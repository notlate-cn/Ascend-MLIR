#pragma once

#include "Runtime/Execution/TaskGraph.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace mlir::runtime {

struct CompilerConfig {
  std::string soc_version;
  std::string arch         = "dav-c220-vec";
  int         opt_level    = 3;
  bool        verbose      = false;
  std::string kernel_type  = "vec";   // "vec" | "cube" | "mix"
};

class Compiler {
public:
  using Config = CompilerConfig;

  explicit Compiler(const Config& cfg = Config{});

  // Retained legacy compile entry point kept only for compatibility coverage
  // and residual non-runtime-native consumers.
  //
  // Compiles src_file into a low-level runtime-consumable artifact path.
  // - vec/cube: returns output_dir/kernel_name.bin
  // - mix: returns a linked device binary path under output_dir
  llvm::Expected<std::string> Compile(const std::string& src_file,
                                      const std::string& output_dir,
                                      const std::string& kernel_name);

private:
  Config cfg_;

  // Run a subprocess synchronously; return non-success Error if exit code != 0.
  llvm::Error RunProcess(const std::vector<std::string>& args);
};

// Retained legacy helper used by the legacy Compiler implementation.
llvm::Error prepareCompileOutputDir(llvm::StringRef outputDir);

// Retained normalization helper used by legacy-oriented tests and compile
// compatibility paths.
KernelArtifact normalizeCompiledArtifact(llvm::StringRef binaryPath,
                                         llvm::StringRef kernelName,
                                         KernelKind kind,
                                         llvm::StringRef socVersion,
                                         llvm::StringRef artifactRoot);

} // namespace mlir::runtime
