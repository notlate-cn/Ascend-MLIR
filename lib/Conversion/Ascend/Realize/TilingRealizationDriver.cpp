//===- TilingRealizationDriver.cpp - Ascend tiling realization --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "TilingRealizationDriver.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::realize;

namespace {

SmallVector<Operation *> collectTileableOps(ModuleOp module) {
  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) {
    if (!op->getAttrOfType<DenseI64ArrayAttr>(kScheduleSelectedTileShapeAttr))
      return;
    if (!isa<TilingInterface>(op))
      return;
    ops.push_back(op);
  });
  return ops;
}

SmallVector<OpFoldResult> buildTileSizes(MLIRContext *ctx,
                                         DenseI64ArrayAttr tileShapeAttr) {
  SmallVector<OpFoldResult> sizes;
  for (int64_t sz : tileShapeAttr.asArrayRef()) {
    sizes.push_back(
        OpFoldResult(IntegerAttr::get(IntegerType::get(ctx, 64), sz)));
  }
  return sizes;
}

} // namespace

namespace mlir::ascend::realize {

LogicalResult TilingRealizationDriver::tileModule(ModuleOp module) const {
  IRRewriter rewriter(module.getContext());

  SmallVector<Operation *> toTile = collectTileableOps(module);
  for (Operation *op : toTile) {
    auto tileShapeAttr =
        op->getAttrOfType<DenseI64ArrayAttr>(kScheduleSelectedTileShapeAttr);

    scf::SCFTilingOptions opts;
    opts.setTileSizes(buildTileSizes(module.getContext(), tileShapeAttr));

    rewriter.setInsertionPoint(op);
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, cast<TilingInterface>(op), opts);
    if (failed(tilingResult))
      return op->emitError("TilingRealizationDriver: scf::tileUsingSCF failed");

    for (Operation *sliceOp : tilingResult->generatedSlices) {
      auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(sliceOp);
      if (!extractSlice)
        continue;
      // Best-effort: no-fusion is safe; the slice is simply left unfused.
      (void)scf::tileAndFuseProducerOfSlice(rewriter, extractSlice,
                                            tilingResult->loops);
    }

    rewriter.replaceOp(op, tilingResult->replacements);
  }
  return success();
}

} // namespace mlir::ascend::realize
