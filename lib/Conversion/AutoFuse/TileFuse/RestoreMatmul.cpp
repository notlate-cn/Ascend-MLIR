#include "Conversion/AutoFuse/AutoFusePasses.h"
#include "TileFuseUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_AUTOFUSERESTOREMATMUL
#define GEN_PASS_DEF_AUTOFUSERESTOREMATMUL
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "auto-fuse-restore-matmul"

using namespace mlir;

namespace mlir::afir {

namespace {

// RestoreMatmul restores only genuine canonical-layout matmuls, so it uses the
// strict form of isMatmulGeneric (shape + body + canonical indexing maps).
// See TileFuseUtils.h.

struct AutoFuseRestoreMatmulPass
    : public ::impl::AutoFuseRestoreMatmulBase<
          AutoFuseRestoreMatmulPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    SmallVector<linalg::GenericOp> targets;
    func.walk([&](linalg::GenericOp op) {
      if (isMatmulGeneric(op, /*checkCanonicalMaps=*/true))
        targets.push_back(op);
    });

    for (linalg::GenericOp gen : targets) {
      builder.setInsertionPoint(gen);
      // Rebuild as linalg.matmul.  Preserve all attrs except those that are
      // intrinsic to the generic form (indexing_maps, iterator_types).
      ValueRange inputs = gen.getDpsInputs();
      ValueRange inits  = gen.getDpsInits();
      Type resTy = gen->getResult(0).getType();
      auto matmul = builder.create<linalg::MatmulOp>(
          gen.getLoc(), TypeRange{resTy}, ValueRange{inputs[0], inputs[1]},
          ValueRange{inits[0]});
      // Copy user-level attrs (`ascendc.unit`, etc.) onto the matmul.
      for (NamedAttribute attr : gen->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name == "indexing_maps" || name == "iterator_types" ||
            name == "operandSegmentSizes")
          continue;
        matmul->setAttr(attr.getName(), attr.getValue());
      }
      gen->getResult(0).replaceAllUsesWith(matmul.getResult(0));
      gen->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAutoFuseRestoreMatmulPass() {
  return std::make_unique<AutoFuseRestoreMatmulPass>();
}

} // namespace mlir::afir
