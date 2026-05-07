//===- SchedulePass.cpp - Ascend V2 schedule pass -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Schedule/SchedulePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DECL_ASCENDSCHEDULEPASS
#define GEN_PASS_DEF_ASCENDSCHEDULEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

struct AscendSchedulePass
    : public ::impl::AscendSchedulePassBase<AscendSchedulePass> {
  using AscendSchedulePassBase::AscendSchedulePassBase;

  void runOnOperation() override {
    ascend::v2::DebugOptions options{
        ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Schedule))
      ascend::v2::emitStageHeader(llvm::errs(),
                                  ascend::v2::DebugStage::Schedule,
                                  getArgument());
  }
};

std::unique_ptr<Pass> createAscendSchedulePass() {
  return std::make_unique<AscendSchedulePass>();
}

} // namespace mlir::afir
