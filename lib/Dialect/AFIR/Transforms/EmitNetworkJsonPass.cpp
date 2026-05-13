//===- EmitNetworkJsonPass.cpp --------------------------------------------===//
//
// Standalone `--emit-network-json=path=...` pass. Used when the user provides
// a hand-written network.mlir (mixed kernel_groupN + __aclnn_xxx calls)
// instead of running --vector-plan-group-outline. Routes through the shared
// NetworkJsonEmitter helper.
//
//===----------------------------------------------------------------------===//
#include "Dialect/AFIR/Transforms/Passes.h"
#include "Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DECL_AFIREMITNETWORKJSONPASS
#define GEN_PASS_DEF_AFIREMITNETWORKJSONPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
struct EmitNetworkJsonPass
    : public mlir::impl::AFIREmitNetworkJsonPassBase<EmitNetworkJsonPass> {
  using Base::Base;

  void runOnOperation() override {
    if (path.empty()) {
      getOperation()->emitError(
          "emit-network-json: --emit-network-json=path=<file> is required");
      return signalPassFailure();
    }
    mlir::func::FuncOp coord;
    getOperation().walk([&](mlir::func::FuncOp f) {
      if (!f.isPrivate()) {
        coord = f;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (!coord) {
      getOperation()->emitError(
          "emit-network-json: no public coordinator func found");
      return signalPassFailure();
    }
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      getOperation()->emitError("emit-network-json: cannot open ")
          << path << ": " << ec.message();
      return signalPassFailure();
    }
    if (auto err =
            mlir::vector_plan::emitNetworkJson(getOperation(), coord, os)) {
      getOperation()->emitError("emit-network-json: ")
          << llvm::toString(std::move(err));
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createAFIREmitNetworkJsonPass() {
  return std::make_unique<EmitNetworkJsonPass>();
}
