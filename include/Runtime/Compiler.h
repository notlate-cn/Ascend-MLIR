// include/Runtime/Compiler.h
#pragma once
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::runtime {

struct CompilerConfig {
  std::string soc_version  = "Ascend910B1";
  std::string arch         = "dav-c220-vec";
  int         opt_level    = 3;
  bool        verbose      = false;
  std::string kernel_type  = "vec";   // "vec" | "cube" | "mix"
};

class Compiler {
public:
  using Config = CompilerConfig;

  explicit Compiler(const Config& cfg = Config{});

  // Compiles src_file → output_dir/kernel_name.bin; returns binary path.
  llvm::Expected<std::string> Compile(const std::string& src_file,
                                      const std::string& output_dir,
                                      const std::string& kernel_name);

private:
  Config cfg_;

  // Run a subprocess synchronously; return non-success Error if exit code != 0.
  llvm::Error RunProcess(const std::vector<std::string>& args);
};

} // namespace mlir::runtime
