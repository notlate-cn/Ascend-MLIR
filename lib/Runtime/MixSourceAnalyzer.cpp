#include "Runtime/MixSourceAnalyzer.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::Expected<MixAnalyzedKernel>
analyzeMixKernel(llvm::StringRef, llvm::StringRef kernelName,
                 llvm::StringRef socVersion) {
  if (kernelName.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "kernelName is required for RuntimeMix mix analysis");
  }

  MixAnalyzedKernel out;
  out.kernelName = kernelName.str();
  out.socVersion = socVersion.str();
  out.launcherSymbol = "aclrtlaunch_" + out.kernelName;
  out.aicEntry = out.kernelName + "_0_mix_aic";
  out.aivEntry = out.kernelName + "_0_mix_aiv";
  out.commonFlags = {
      "-c",
      "-x",
      "cce",
      "-O3",
      "--cce-aicore-only",
      "-std=c++17",
      "--cce-disable-kernel-global-attr-check",
      "-mllvm",
      "-cce-aicore-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-function-stack-size=0x8000",
      "-mllvm",
      "-cce-aicore-dcci-insert-for-scalar=false",
      "-DASCENDC_DUMP=0",
      "-D__NPU_TILING__",
      "-DTILING_KEY_VAR=0",
  };
  out.aicDefines = {
      "__MIX_CORE_MACRO__=1",
      "auto_gen_" + out.kernelName + "_kernel=" + out.aicEntry,
      "__DAV_C220_CUBE__",
  };
  out.aivDefines = {
      "__MIX_CORE_MACRO__=1",
      "auto_gen_" + out.kernelName + "_kernel=" + out.aivEntry,
      "__DAV_C220_VEC__",
  };
  return out;
}

} // namespace mlir::runtime
