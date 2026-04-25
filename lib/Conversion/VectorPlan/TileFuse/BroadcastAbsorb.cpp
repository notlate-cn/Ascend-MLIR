// BroadcastAbsorb.cpp — Phase 1 Collapse: absorb linalg.broadcast into
// downstream linalg.generic indexing maps.
//
// Stub — implementation forthcoming.

#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::afir {

namespace {
struct VectorPlanBroadcastAbsorbPass
    : public PassWrapper<VectorPlanBroadcastAbsorbPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorPlanBroadcastAbsorbPass)

  StringRef getArgument() const override {
    return "vector-plan-broadcast-absorb";
  }
  StringRef getDescription() const override {
    return "Absorb linalg.broadcast into downstream linalg.generic indexing maps";
  }

  void runOnOperation() override {
    // TODO: Phase 1 Collapse implementation.
  }
};
} // namespace

std::unique_ptr<Pass> createVectorPlanBroadcastAbsorbPass() {
  return std::make_unique<VectorPlanBroadcastAbsorbPass>();
}

} // namespace mlir::afir
