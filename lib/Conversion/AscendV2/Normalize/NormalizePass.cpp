//===- NormalizePass.cpp - Ascend V2 normalize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Normalize/NormalizePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DECL_ASCENDNORMALIZEPASS
#define GEN_PASS_DEF_ASCENDNORMALIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

struct AscendNormalizePass
    : public ::impl::AscendNormalizePassBase<AscendNormalizePass> {
  using AscendNormalizePassBase::AscendNormalizePassBase;

  void runOnOperation() override {
    ascend::v2::DebugOptions options{
        ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Normalize))
      ascend::v2::emitStageHeader(llvm::errs(),
                                  ascend::v2::DebugStage::Normalize,
                                  getArgument());
  }
};

std::unique_ptr<Pass> createAscendNormalizePass() {
  return std::make_unique<AscendNormalizePass>();
}

} // namespace mlir::afir
