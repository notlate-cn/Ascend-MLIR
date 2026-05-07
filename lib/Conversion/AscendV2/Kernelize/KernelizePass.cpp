//===- KernelizePass.cpp - Ascend V2 kernelize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/KernelizePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DECL_ASCENDKERNELIZEPASS
#define GEN_PASS_DEF_ASCENDKERNELIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

struct AscendKernelizePass
    : public ::impl::AscendKernelizePassBase<AscendKernelizePass> {
  using AscendKernelizePassBase::AscendKernelizePassBase;

  void runOnOperation() override {
    ascend::v2::DebugOptions options{
        ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize))
      ascend::v2::emitStageHeader(llvm::errs(),
                                  ascend::v2::DebugStage::Kernelize,
                                  getArgument());
  }
};

std::unique_ptr<Pass> createAscendKernelizePass() {
  return std::make_unique<AscendKernelizePass>();
}

} // namespace mlir::afir
