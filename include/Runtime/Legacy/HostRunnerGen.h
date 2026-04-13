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
    std::string soc_version;
    int         num_inputs  = 1;
    int         num_outputs = 1;
    // Per-output dtype strings used by --output-dtype<N> flags in the runner.
    // Values: "f16" | "bf16" | "f32" | "i8" | "i32" | "i64".
    // If empty or shorter than num_outputs, missing entries default to "f16".
    std::vector<std::string> output_dtypes;
    std::vector<std::string> tiling_layout;  // e.g. {"int64","int64","int32"}
    size_t      workspace_size = 8192;       // bytes allocated for workspace GM buffer
    bool        verbose = false;
    // Note: arch is NOT needed here. runner is a Host (x86/aarch64) executable
    // compiled with g++. Only the kernel .bin is bisheng-compiled for the NPU arch.
  };

  // Generate runner.cpp and compile it to output_dir/runner.
  // Returns path to the runner executable on success.
  // The runner accepts at runtime:
  //   --bin <kernel.bin>              (required)
  //   --tiling-params "K1=V1,..."     (optional)
  //   --tiling-layout "int64,..."     (optional, overrides compiled-in layout)
  //   --inputs a.npy,b.npy            (required)
  //   --output<N> out.npy             (optional, one per output, default /dev/null)
  //   --output-shape<N> "D0,D1,..."   (optional, defaults to inputs[0] shape)
  //   --output-dtype<N> f16|bf16|f32|i8|i32|i64  (optional, overrides compiled-in default)
  //   --block-dim N                   (optional, default 1)
  llvm::Expected<std::string> Generate(const Config& cfg,
                                        const std::string& output_dir);
};

} // namespace mlir::runtime
