#include "Conversion/TorchFrontend/TorchFrontendPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_REMOVECFASSERT
#define GEN_PASS_DEF_REMOVECFASSERT
#include "Conversion/TorchFrontend/TorchFrontendPasses.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

struct RemoveCfAssertPass
    : public ::impl::RemoveCfAssertBase<RemoveCfAssertPass> {
  void runOnOperation() override {
    getOperation().walk([](cf::AssertOp op) { op.erase(); });
  }
};

} // namespace

std::unique_ptr<Pass> createRemoveCfAssertPass() {
  return std::make_unique<RemoveCfAssertPass>();
}

} // namespace mlir::afir