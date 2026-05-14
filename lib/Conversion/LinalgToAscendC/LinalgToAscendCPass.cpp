/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under terms and conditions of the CANN Open Software License Agreement
 * Version 2.0 (the "License"). Please refer to LICENSE in the root of the
 * software repository for the full text of the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#include "Conversion/LinalgToAscendC/LinalgToAscendCPass.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#define GEN_PASS_DECL_LINALGTOASCENDCPASS
#define GEN_PASS_DEF_LINALGTOASCENDCPASS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "linalg-to-ascendc"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

//===----------------------------------------------------------------------===//
// AscendCBufferContext helpers (definitions)
//===----------------------------------------------------------------------===//

int64_t getMemorySpace(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return -1;
  Attribute space = memrefType.getMemorySpace();
  if (!space)
    return 0;
  if (auto intAttr = dyn_cast<IntegerAttr>(space))
    return intAttr.getInt();
  return -1;
}

static bool hasOnChipOutput(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumDpsInits() == 0)
    return false;
  int64_t memorySpace =
      getMemorySpace(linalgOp.getDpsInitOperand(0)->get().getType());
  return memorySpace > 0;
}

static bool isRank2SwapPermutation(ArrayRef<int64_t> permutation) {
  return permutation.size() == 2 && permutation[0] == 1 &&
         permutation[1] == 0;
}

static bool isValidPermutation(ArrayRef<int64_t> permutation) {
  SmallVector<bool, 8> seen(permutation.size(), false);
  for (int64_t position : permutation) {
    if (position < 0 || position >= static_cast<int64_t>(permutation.size()))
      return false;
    if (seen[position])
      return false;
    seen[position] = true;
  }
  return true;
}

FailureOr<TransposeLoweringSpec>
buildTransposeLoweringSpec(linalg::TransposeOp transpose) {
  TransposeLoweringSpec spec;
  spec.rank = transpose.getPermutation().size();
  llvm::append_range(spec.permutation, transpose.getPermutation());
  spec.hasOnChipOutput = hasOnChipOutput(transpose);
  return spec;
}

FailureOr<TransposeLoweringSpec>
buildTransposeLoweringSpec(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return failure();
  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != 2)
    return failure();

  AffineMap inMap = maps[0];
  AffineMap outMap = maps[1];
  unsigned rank = generic.getIteratorTypesArray().size();
  if (!outMap.isIdentity() || inMap.getNumResults() != rank)
    return failure();

  TransposeLoweringSpec spec;
  spec.rank = rank;
  spec.hasOnChipOutput = hasOnChipOutput(generic);
  llvm::SmallSet<unsigned, 8> seen;
  for (unsigned r = 0; r < rank; ++r) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(r));
    if (!dimExpr)
      return failure();
    unsigned position = dimExpr.getPosition();
    if (position >= rank || !seen.insert(position).second)
      return failure();
    spec.permutation.push_back(position);
  }

  Block &body = *generic.getBody();
  if (body.getOperations().size() != 1)
    return failure();
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body.front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return failure();
  auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  if (!blockArg || blockArg.getArgNumber() != 0)
    return failure();
  return spec;
}

TransposeLoweringPlan planTransposeLowering(const TransposeLoweringSpec &spec) {
  TransposeLoweringPlan plan;
  if (spec.hasOnChipOutput && isRank2SwapPermutation(spec.permutation)) {
    plan.kind = TransposeLoweringKind::AscendCSimple2D;
    return plan;
  }
  if (!spec.hasOnChipOutput && isValidPermutation(spec.permutation))
    plan.kind = TransposeLoweringKind::ScalarMemRefLoop;
  return plan;
}

Value computeElementCount(OpBuilder &b, Location loc, Value memrefVal) {
  auto memrefType = cast<MemRefType>(memrefVal.getType());
  ArrayRef<int64_t> shape = memrefType.getShape();
  Value count;
  for (int64_t i = 0, rank = (int64_t)shape.size(); i < rank; ++i) {
    Value dimVal;
    if (ShapedType::isDynamic(shape[i]))
      dimVal = b.create<memref::DimOp>(loc, memrefVal, i);
    else
      dimVal = b.create<arith::ConstantIndexOp>(loc, shape[i]);
    count = count ? b.create<arith::MulIOp>(loc, count, dimVal) : dimVal;
  }
  if (!count)
    count = b.create<arith::ConstantIndexOp>(loc, 1);
  return count;
}

static Value getEnclosingLoopStepBound(Value value, Operation *anchor) {
  auto matchesEnclosingStep = [&](Value candidate) -> bool {
    for (Operation *parent = anchor; parent; parent = parent->getParentOp()) {
      auto forOp = dyn_cast<scf::ForOp>(parent);
      if (forOp && candidate == forOp.getStep())
        return true;
    }
    return false;
  };

  if (auto minOp = value.getDefiningOp<arith::MinSIOp>()) {
    if (matchesEnclosingStep(minOp.getLhs()))
      return minOp.getLhs();
    if (matchesEnclosingStep(minOp.getRhs()))
      return minOp.getRhs();
  }
  if (auto minOp = value.getDefiningOp<arith::MinUIOp>()) {
    if (matchesEnclosingStep(minOp.getLhs()))
      return minOp.getLhs();
    if (matchesEnclosingStep(minOp.getRhs()))
      return minOp.getRhs();
  }
  if (auto minOp = value.getDefiningOp<affine::AffineMinOp>()) {
    for (Value operand : minOp.getOperands())
      if (matchesEnclosingStep(operand))
        return operand;
  }
  return value;
}

static Value getAllocDynamicSizeBound(OpBuilder &b, Location loc, Value size,
                                      Operation *anchor) {
  if (auto dimOp = size.getDefiningOp<memref::DimOp>()) {
    auto subviewOp = dimOp.getSource().getDefiningOp<memref::SubViewOp>();
    auto dimIndexOp = dimOp.getIndex().getDefiningOp<arith::ConstantIndexOp>();
    if (subviewOp && dimIndexOp) {
      unsigned dim = static_cast<unsigned>(dimIndexOp.value());
      SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
      if (dim < mixedSizes.size()) {
        OpFoldResult subviewSize = mixedSizes[dim];
        if (auto attr = subviewSize.dyn_cast<Attribute>())
          return b.create<arith::ConstantIndexOp>(
              loc, cast<IntegerAttr>(attr).getInt());
        return getEnclosingLoopStepBound(subviewSize.get<Value>(), anchor);
      }
    }
  }
  return getEnclosingLoopStepBound(size, anchor);
}

Value computeByteCount(OpBuilder &b, Location loc, Value memrefVal) {
  auto memrefType = cast<MemRefType>(memrefVal.getType());
  unsigned bytesPerElem = memrefType.getElementTypeBitWidth() / 8;
  Value elemCount = computeElementCount(b, loc, memrefVal);
  Value bytesPerElemVal = b.create<arith::ConstantIndexOp>(loc, bytesPerElem);
  return b.create<arith::MulIOp>(loc, elemCount, bytesPerElemVal);
}

// Compute byte count for an AllocOp using its dynamic size operands directly
// (avoids inserting memref.dim ops; the alloc's dynamic sizes are already
// the correct SSA values and dominate the same scope as the alloc).
Value computeAllocByteCount(OpBuilder &b, Location loc,
                             memref::AllocOp allocOp) {
  auto memrefType = allocOp.getType();
  ArrayRef<int64_t> shape = memrefType.getShape();
  unsigned bytesPerElem = memrefType.getElementTypeBitWidth() / 8;
  bool preserveExactTailSize =
      static_cast<TPosition>(getMemorySpace(memrefType)) == TPosition::VECOUT &&
      memrefType.getRank() == 1;
  Value count;
  unsigned dynIdx = 0;
  for (int64_t dim : shape) {
    Value dimVal;
    if (ShapedType::isDynamic(dim)) {
      Value rawDim = allocOp.getDynamicSizes()[dynIdx++];
      dimVal = preserveExactTailSize
                   ? rawDim
                   : getAllocDynamicSizeBound(b, loc, rawDim,
                                              allocOp.getOperation());
    } else {
      dimVal = b.create<arith::ConstantIndexOp>(loc, dim);
    }
    count = count ? b.create<arith::MulIOp>(loc, count, dimVal) : dimVal;
  }
  if (!count)
    count = b.create<arith::ConstantIndexOp>(loc, 1);
  Value bytesPerElemVal = b.create<arith::ConstantIndexOp>(loc, bytesPerElem);
  return b.create<arith::MulIOp>(loc, count, bytesPerElemVal);
}

/// Walk through subviews and scf.for iter_args to find the ultimate source.
static Value resolveToAllocRoot(Value v) {
  const int maxDepth = 20;
  for (int i = 0; i < maxDepth; ++i) {
    if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
      v = subview.getSource();
      continue;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (forOp && blockArg.getArgNumber() > 0) {
        unsigned iterIdx = blockArg.getArgNumber() - 1;
        if (iterIdx < forOp.getInitArgs().size()) {
          v = forOp.getInitArgs()[iterIdx];
          continue;
        }
      }
    }
    break;
  }
  return v;
}

Value AscendCBufferContext::getQueue(Value memref) const {
  Value root = resolveToAllocRoot(memref);
  auto it = allocToQueue.find(root);
  if (it != allocToQueue.end())
    return it->second;
  return {};
}

Value AscendCBufferContext::getTBuf(Value memref) const {
  Value root = resolveToAllocRoot(memref);
  auto it = allocToTBuf.find(root);
  if (it != allocToTBuf.end())
    return it->second;
  return {};
}

Value AscendCBufferContext::getLiveTensor(Value memref) const {
  Value root = resolveToAllocRoot(memref);
  auto it = allocToLiveTensor.find(root);
  if (it != allocToLiveTensor.end())
    return it->second;
  return {};
}

static void eraseDeadTBufInitializers(func::FuncOp funcOp) {
  SmallVector<TBufOp> deadTBufs;
  funcOp.walk([&](TBufOp tbuf) {
    bool onlyInitUsers = true;
    for (Operation *user : tbuf->getUsers()) {
      if (!isa<TPipeInitBufferOp>(user)) {
        onlyInitUsers = false;
        break;
      }
    }
    if (onlyInitUsers)
      deadTBufs.push_back(tbuf);
  });

  for (TBufOp tbuf : deadTBufs) {
    SmallVector<Operation *> users(tbuf->getUsers().begin(),
                                   tbuf->getUsers().end());
    for (Operation *user : users)
      if (isa<TPipeInitBufferOp>(user))
        user->erase();
    if (tbuf->use_empty())
      tbuf.erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass: build context, run data-move and compute conversions
//===----------------------------------------------------------------------===//

LogicalResult lowerLinalgToAscendC(func::FuncOp funcOp) {
  if (funcOp.isExternal())
    return success();

  MLIRContext *ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  if (failed(materializeSelectedReductionTiles(funcOp)))
    return failure();
  if (failed(materializeSelectedAllParallelTiles(funcOp)))
    return failure();

  // -----------------------------------------------------------------------
  // Phase 0: Build the shared pipe + one queue per on-chip alloc.
  //
  // Rules:
  //  - Exactly ONE PipeOp per function, inserted at the top of entry block.
  //  - One QueueOp per on-chip memref.alloc (memory_space > 0).
  //  - TBuf + TPipeInitBufferOp inserted right after the alloc that owns it,
  //    so that dynamic memref.dim ops remain dominated by the alloc.
  // -----------------------------------------------------------------------
  Block &entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);
  Value pipe = builder.create<PipeOp>(funcOp.getLoc(), PipeType::get(ctx));

  // lastQueueInserted is used to keep all QueueOps together at the top of
  // the entry block, right after pipe (for readability).
  Value lastQueueInserted = pipe;

  AscendCBufferContext bufCtx;
  bufCtx.pipe = pipe;

  funcOp.walk([&](memref::AllocOp allocOp) {
    int64_t ms = getMemorySpace(allocOp.getType());
    if (ms <= 0)
      return;

    auto pos = static_cast<TPosition>(ms);

    // Create QueueOp right after the previous queue (entry block top).
    builder.setInsertionPointAfterValue(lastQueueInserted);
    Value queue =
        builder.create<QueueOp>(allocOp.getLoc(), QueueType::get(ctx, pos, 1));
    lastQueueInserted = queue;

    // TBuf + init_buffer inserted right before the alloc.
    // len is computed from the alloc's dynamic size operands (its inputs),
    // which are defined before the alloc and dominate the same scope.
    // HoistQueBindPass will then lift tbuf+init_buffer to the entry block
    // whenever all operands dominate the enclosing loop.
    builder.setInsertionPoint(allocOp);
    Value tbuf =
        builder.create<TBufOp>(allocOp.getLoc(), TBufType::get(ctx, pos));
    Value len = computeAllocByteCount(builder, allocOp.getLoc(), allocOp);
    builder.create<TPipeInitBufferOp>(allocOp.getLoc(), pipe, tbuf, len);
    // Initialize the TQue so that AllocTensor returns a tensor with a
    // valid GetSize().  Without this, ReduceSum2DL2 computes a division
    // by zero (_afir_cols = accumLt.GetSize() / vecoutLt.GetSize()).
    Value depth = builder.create<arith::ConstantOp>(
        allocOp.getLoc(), builder.getI32IntegerAttr(1));
    builder.create<TPipeInitQueueOp>(allocOp.getLoc(), pipe, queue, depth, len);

    bufCtx.allocToQueue[allocOp.getResult()] = queue;
    bufCtx.allocToTBuf[allocOp.getResult()] = tbuf;
  });

  // -----------------------------------------------------------------------
  // Phase 1: Convert memref.copy → AscendC data-move ops.
  // -----------------------------------------------------------------------
  if (failed(convertDataMove(funcOp, bufCtx)))
    return failure();

  // -----------------------------------------------------------------------
  // Phase 2: Convert linalg compute ops → AscendC compute ops.
  // -----------------------------------------------------------------------
  if (failed(convertCompute(funcOp, bufCtx)))
    return failure();
  eraseDeadTBufInitializers(funcOp);

  // -----------------------------------------------------------------------
  // Phase 3: Hoist pipe/queue/tbuf/init_buffer to entry block wherever
  // all operands dominate the enclosing loop (i.e., static tile sizes or
  // function-argument-based sizes).
  // -----------------------------------------------------------------------
  RewritePatternSet hoistPatterns(ctx);
  hoistPatterns.add<ascendc::HoistOpPattern<arith::ConstantOp>,
                    ascendc::HoistOpPattern<arith::MulIOp>,
                    ascendc::HoistOpPattern<ascendc::QueueOp>,
                    ascendc::HoistOpPattern<ascendc::TBufOp>,
                    ascendc::HoistOpPattern<ascendc::TPipeInitBufferOp>,
                    ascendc::HoistOpPattern<ascendc::TPipeInitQueueOp>>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(hoistPatterns))))
    return failure();

  return success();
}

namespace {
struct LinalgToAscendCPass
    : public ::impl::LinalgToAscendCPassBase<LinalgToAscendCPass> {

  void runOnOperation() override {
    if (failed(lowerLinalgToAscendC(getOperation()))) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "=== After LinalgToAscendCPass ===\n");
    LLVM_DEBUG(getOperation().print(llvm::dbgs()));
  }
};
} // namespace

std::unique_ptr<Pass> createLinalgToAscendCPass() {
  return std::make_unique<LinalgToAscendCPass>();
}

} // namespace afir
} // namespace mlir
