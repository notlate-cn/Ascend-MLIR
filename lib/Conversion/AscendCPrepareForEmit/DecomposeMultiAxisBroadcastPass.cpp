#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DECL_ASCENDCDECOMPOSEMULTIAXISBROADCASTPASS
#define GEN_PASS_DEF_ASCENDCDECOMPOSEMULTIAXISBROADCASTPASS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "ascendc-decompose-multi-axis-broadcast"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir::afir {

namespace {

// Returns true if `v` is the result of an arith.constant integer with value 1.
static bool isStaticOne(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt() == 1;
  return false;
}

// Locate the unique PipeOp in `func`.  LinalgToAscendC creates exactly one at
// the top of the entry block; returns nullptr if missing.
static Value findPipe(func::FuncOp func) {
  Value pipe;
  func.walk([&](PipeOp op) {
    pipe = op.getResult();
    return WalkResult::interrupt();
  });
  return pipe;
}

// Allocate a fresh VECCALC LocalTensor sized by the (i32) shape values.
// Mirrors `allocVeccalc` in ComputeConversion.cpp but inserts at `b`'s
// current insertion point.
static Value allocVeccalcTensor(OpBuilder &b, Location loc, Value pipe,
                                 Type elemType, ArrayRef<Value> shapeI32) {
  // total elements = product(shapeI32 cast to index)
  Value totalElems;
  for (Value s : shapeI32) {
    Value sIdx =
        b.create<arith::IndexCastOp>(loc, b.getIndexType(), s);
    totalElems =
        totalElems ? b.create<arith::MulIOp>(loc, totalElems, sIdx) : sIdx;
  }
  if (!totalElems)
    totalElems = b.create<arith::ConstantIndexOp>(loc, 1);
  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value byteSize = b.create<arith::MulIOp>(
      loc, totalElems, b.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value tbuf = b.create<TBufOp>(loc, TBufType::get(b.getContext(),
                                                    TPosition::VECCALC));
  b.create<TPipeInitBufferOp>(loc, pipe, tbuf, byteSize);
  return b.create<TBufGetTensorOp>(loc, LocalTensorType::get(elemType), tbuf,
                                    /*len=*/Value{});
}

// Decompose a multi-axis BroadcastL2Op into a chain of single-axis ones.
// Returns true if `op` was decomposed (and erased); false if it was already
// single-axis (left untouched).
static bool decomposeBroadcast(BroadcastL2Op op, Value pipe) {
  uint32_t rank = op.getConstRank();
  auto srcShape = op.getSrcShape();
  auto dstShape = op.getDstShape();

  SmallVector<int> bcastAxes;
  for (uint32_t i = 0; i < rank; ++i) {
    if (isStaticOne(srcShape[i]) && !isStaticOne(dstShape[i]))
      bcastAxes.push_back(static_cast<int>(i));
  }
  if (bcastAxes.size() <= 1)
    return false;

  OpBuilder b(op);
  Location loc = op.getLoc();
  Type elemType =
      cast<LocalTensorType>(op.getDst().getType()).getElementType();

  // Broadcast one axis at a time.  At each step pick a *foldable* axis — one
  // whose prefix dims [0,ba) are all currently statically-1 OR whose suffix
  // dims (ba, N) are all currently statically-1.  This avoids leaving a
  // middle-axis broadcast for the CannTranslation fold to handle when both
  // prefix and suffix have already grown non-1 (which is not foldable to a
  // single 2D AscendC::Broadcast).
  //
  // Intermediate steps write to fresh VECCALC tensors; the last step writes
  // to the original dst so existing users of op.getDst() see the final result.
  Value curTensor = op.getSrc();
  SmallVector<Value> curShape(srcShape.begin(), srcShape.end());
  SmallVector<int> remaining = bcastAxes;
  while (!remaining.empty()) {
    // An axis `ba` is foldable in `shape` when its prefix [0,ba) OR its suffix
    // (ba,rank) is all statically-1 (-> a single 2D row/col AscendC::Broadcast).
    auto foldableIn = [&](int ba, ArrayRef<Value> shape) {
      auto allOneIn = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i)
          if (!isStaticOne(shape[i]))
            return false;
        return true;
      };
      return allOneIn(0, ba) || allOneIn(ba + 1, rank);
    };
    // Greedily pick a foldable axis, but with 1-step lookahead: broadcasting an
    // axis grows its dim to non-1, which can strand a remaining axis (e.g. for
    // [1,1,N]->[D0,D1,N], broadcasting outer axis 0 first leaves axis 1 with a
    // non-1 prefix AND suffix).  Prefer the foldable axis that keeps the most
    // remaining axes foldable; this picks the inner axis first there.
    int pickIdx = -1, bestScore = -1;
    for (int k = 0; k < (int)remaining.size(); ++k) {
      int ba = remaining[k];
      if (!foldableIn(ba, curShape))
        continue;
      SmallVector<Value> hyp(curShape.begin(), curShape.end());
      hyp[ba] = dstShape[ba]; // becomes non-1 after broadcasting
      int score = 0;
      for (int j = 0; j < (int)remaining.size(); ++j)
        if (j != k && foldableIn(remaining[j], hyp))
          ++score;
      if (score > bestScore) {
        bestScore = score;
        pickIdx = k;
      }
    }
    if (pickIdx < 0) {
      op.emitOpError(
          "multi-axis broadcast with no foldable order: no remaining axis has "
          "an all-const-1 prefix or suffix in the current shape");
      return false;
    }
    int ax = remaining[pickIdx];
    bool isLast = (remaining.size() == 1);

    SmallVector<Value> nextShape = curShape;
    nextShape[ax] = dstShape[ax];

    Value nextTensor = isLast ? op.getDst()
                              : allocVeccalcTensor(b, loc, pipe, elemType,
                                                    nextShape);

    auto stepOp = b.create<BroadcastL2Op>(
        loc, nextTensor, curTensor, /*dstShape=*/nextShape,
        /*srcShape=*/curShape,
        b.getI32IntegerAttr(static_cast<int32_t>(rank)));
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName().getValue().starts_with("ascendc."))
        stepOp->setAttr(attr.getName(), attr.getValue());
    }

    curTensor = nextTensor;
    curShape = nextShape;
    remaining.erase(remaining.begin() + pickIdx);
  }

  op.erase();
  return true;
}

struct AscendCDecomposeMultiAxisBroadcastPass
    : public ::impl::AscendCDecomposeMultiAxisBroadcastPassBase<
          AscendCDecomposeMultiAxisBroadcastPass> {
  using AscendCDecomposeMultiAxisBroadcastPassBase::
      AscendCDecomposeMultiAxisBroadcastPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect first, then transform — walk-and-erase is unsafe.
    SmallVector<BroadcastL2Op> targets;
    func.walk([&](BroadcastL2Op op) { targets.push_back(op); });
    if (targets.empty())
      return;

    Value pipe = findPipe(func);
    if (!pipe) {
      func.emitOpError(
          "ascendc-decompose-multi-axis-broadcast: no ascendc.pipe found; "
          "expected LinalgToAscendC to have created one");
      signalPassFailure();
      return;
    }

    for (BroadcastL2Op op : targets)
      (void)decomposeBroadcast(op, pipe);
  }
};

} // namespace

std::unique_ptr<Pass> createAscendCDecomposeMultiAxisBroadcastPass() {
  return std::make_unique<AscendCDecomposeMultiAxisBroadcastPass>();
}

} // namespace mlir::afir
