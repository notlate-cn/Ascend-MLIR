//===- MarkStructuredOpsPass.cpp - Mark structured linalg ops -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GEN_PASS_DECL_MARKSTRUCTUREDOPSPASS
#define GEN_PASS_DEF_MARKSTRUCTUREDOPSPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

struct MarkStructuredOpsPass
    : public ::impl::MarkStructuredOpsPassBase<MarkStructuredOpsPass> {
  void runOnOperation() override;
};

// Five-condition gather detection.
// Returns {isGather, gatherDim, isEmbedding} where gatherDim is the dimension
// of the data tensor used as the dynamic gather axis.
//
// Conditions (all must hold):
//  1. Body contains at least one tensor.extract op.
//  2. The tensor operand of extract is either:
//       (a) a block argument from ins, OR
//       (b) a value captured from outside the body (e.g. another generic's result).
//  3. One ins block arg is used as a dynamic index into the extract at
//     exactly one dimension position.
//  4. That indices ins has a lower-rank affine map (broadcast map).
//  5. All other index operands of the extract come from linalg.index ops.
//
// After conditions pass:
//  - Inspect indices affine map: if indices vary along d0 → embedding (row),
//    otherwise → index_select (column).
struct GatherInfo {
  bool isGather = false;
  int64_t gatherDim = -1;   // dimension in data tensor used as gather axis
  bool isEmbedding = false; // true = row selection, false = column selection
};

GatherInfo detectGather(linalg::GenericOp op) {
  GatherInfo info;
  unsigned iterRank = op.getIteratorTypesArray().size();

  // Must have at least 1 ins and 1 out.
  // (data tensor may be captured externally rather than passed as an ins arg)
  if (op.getNumDpsInputs() < 1 || op.getNumDpsInits() != 1)
    return info;

  // Condition 1: body contains tensor.extract.
  tensor::ExtractOp extractOp;
  op.getBody()->walk([&](tensor::ExtractOp e) {
    if (!extractOp)
      extractOp = e;
  });
  if (!extractOp)
    return info;

  // Condition 2: extracted tensor is either:
  //   (a) a block argument in ins, OR
  //   (b) a value captured from outside the linalg.generic body.
  // In case (a), track its arg index so we don't confuse it with the indices
  // block arg. In case (b), any block arg used as a dynamic index is the
  // indices tensor.
  int dataArgIdx = -1; // -1 means captured (case b)
  if (auto dataBa = dyn_cast<BlockArgument>(extractOp.getTensor())) {
    if (dataBa.getOwner() == op.getBody()) {
      // Case (a): block arg inside the linalg body — must be an ins arg.
      if ((int)dataBa.getArgNumber() >= op.getNumDpsInputs())
        return info;
      dataArgIdx = (int)dataBa.getArgNumber();
    }
    // else: case (b) — func arg or outer-block arg captured into the region.
  }
  // (case b) captured tensor: getDefiningOp may exist or it's a func/outer arg
  // — both are fine; we just can't check further here.

  // Conditions 3 & 5: scan extract indices.
  // Exactly one index must come from an ins block arg (the indices tensor).
  // All remaining indices must come from linalg.index ops.
  int indicesArgIdx = -1;
  int64_t dynamicDim = -1;

  for (auto [dimPos, idxVal] : llvm::enumerate(extractOp.getIndices())) {
    // Peel through arith.index_cast to get the underlying value.
    Value baseVal = idxVal;
    if (auto castOp = idxVal.getDefiningOp<arith::IndexCastOp>())
      baseVal = castOp.getIn();

    if (auto ba = dyn_cast<BlockArgument>(baseVal)) {
      if (ba.getOwner() != op.getBody())
        return info;
      if ((int)ba.getArgNumber() == dataArgIdx)
        return info;
      if ((int)ba.getArgNumber() >= op.getNumDpsInputs())
        return info;
      if (indicesArgIdx != -1)
        return info; // more than one dynamic dim — not simple gather
      indicesArgIdx = (int)ba.getArgNumber();
      dynamicDim = (int64_t)dimPos;
    } else if (baseVal.getDefiningOp<linalg::IndexOp>() ||
               idxVal.getDefiningOp<linalg::IndexOp>()) {
      // Condition 5: OK
    } else {
      return info; // unexpected value source
    }
  }
  if (indicesArgIdx == -1 || dynamicDim == -1)
    return info;

  // Condition 4: indices ins has a lower-rank affine map (broadcast map).
  auto maps = op.getIndexingMapsArray();
  AffineMap indicesMap = maps[(unsigned)indicesArgIdx];
  if (indicesMap.getNumResults() >= iterRank)
    return info;

  // Only support 1D indices for now.
  if (indicesMap.getNumResults() != 1)
    return info;

  auto dimExpr = dyn_cast<AffineDimExpr>(indicesMap.getResult(0));
  if (!dimExpr)
    return info;

  info.isGather = true;
  info.gatherDim = dynamicDim;
  // Row selection (embedding): indices vary along d0.
  info.isEmbedding = (dimExpr.getPosition() == 0);
  return info;
}

}  // namespace

void MarkStructuredOpsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp.getContext());

  funcOp.walk([&](linalg::GenericOp op) {
    GatherInfo info = detectGather(op);
    if (!info.isGather)
      return;

    if (info.isEmbedding) {
      op->setAttr("embedding_dim",
                  builder.getI64IntegerAttr(info.gatherDim));
    } else {
      op->setAttr("gather_dim",
                  builder.getI64IntegerAttr(info.gatherDim));
    }
  });
}

std::unique_ptr<Pass> createMarkStructuredOpsPass() {
  return std::make_unique<MarkStructuredOpsPass>();
}

}  // namespace mlir::afir
