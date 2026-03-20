// include/Runtime/HostRunnerGen.h
#pragma once
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace mlir::runtime {

class HostRunnerGen {
public:
  struct Config {
    std::string kernel_name;
    std::string kernel_type;  // "vec" | "cube" | "mix"
    // kernel_type → DevBinary magic:
    //   "vec"  → 0x41415246 (MAGIC_ELF_AIVEC)
    //   "cube" → 0x41494343 (MAGIC_ELF_AICUBE)
    //   "mix"  → 0x41415246 (same as vec, conservative default)
    std::string soc_version = "Ascend910B1";
    int         num_inputs  = 1;
    int         num_outputs = 1;  // currently only 1 is supported
    std::vector<std::string> tiling_layout;  // e.g. {"int64","int64","int32"}
    bool        verbose = false;
    // Note: arch is NOT needed here. runner is a Host (x86/aarch64) executable
    // compiled with g++. Only the kernel .bin is bisheng-compiled for the NPU arch.
  };

  // Generate runner.cpp and compile it to output_dir/runner.
  // Returns path to the runner executable on success.
  // Returns error if num_outputs != 1 (not yet supported).
  // The runner accepts at runtime:
  //   --bin <kernel.bin>           (required)
  //   --tiling-params "K1=V1,..."  (optional)
  //   --tiling-layout "int64,..."  (optional, overrides compiled-in layout)
  //   --inputs a.npy,b.npy         (required)
  //   --output out.npy             (optional, default /dev/null for perf-only runs)
  //   --block-dim N                (optional, default 1)
  llvm::Expected<std::string> Generate(const Config& cfg,
                                        const std::string& output_dir);
};

} // namespace mlir::runtime
