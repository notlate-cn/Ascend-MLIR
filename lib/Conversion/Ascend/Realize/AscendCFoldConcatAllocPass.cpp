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

// AscendCFoldConcatAllocPass
//
// Problem: after bufferization of decomposed tensor.concat, the IR has:
//
//   %alloc_c   = memref.alloc(%M, %N)
//   %alloc_d   = memref.alloc(%M, %N)
//   %2 = scf.for ... iter_args(%a = %alloc_c) { ... writes %a ... }
//   %3 = scf.for ... iter_args(%a = %alloc_d) { ... writes %a ... }
//   %alloc_out = memref.alloc(%2M, %N)
//   %sub0 = memref.subview %alloc_out[0,0][M,N]
//   memref.copy %2, %sub0         // ← %2 is the for result aliasing %alloc_c
//   %sub1 = memref.subview %alloc_out[M,0][M,N]
//   memref.copy %3, %sub1
//   return %alloc_out
//
// This pass reconstructs alloc_out and subviews before the loops using the
// srcAlloc sizes (which are loop-independent), then redirects each for loop's
// iter_arg init to write directly into the subview.  The copies are removed.
//
// After the pass:
//   %alloc_out = memref.alloc(%2M, %N)      <- moved before loops
//   %sub0 = memref.subview %alloc_out[0,0][M,N]
//   %sub1 = memref.subview %alloc_out[M,0][M,N]
//   scf.for ... iter_args(%a = %sub0_cast) { ... writes %a ... }
//   scf.for ... iter_args(%a = %sub1_cast) { ... writes %a ... }
//   return %alloc_out

#include "Conversion/Ascend/Realize/AscendCFoldConcatAllocPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DECL_ASCENDCFOLDCONCATALLOCPASS
#define GEN_PASS_DEF_ASCENDCFOLDCONCATALLOCPASS
#include "Conversion/Ascend/Passes.h.inc"

#define DEBUG_TYPE "ascendc-fold-concat-alloc"

using namespace mlir;

namespace mlir::afir {

struct AscendCFoldConcatAllocPass
    : public ::impl::AscendCFoldConcatAllocPassBase<AscendCFoldConcatAllocPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Find the single return op.
    func::ReturnOp retOp;
    func.walk([&](func::ReturnOp op) { retOp = op; });
    if (!retOp || retOp.getNumOperands() != 1)
      return;

    // Return value must be a memref.alloc.
    Value retVal = retOp.getOperand(0);
    auto outAlloc = retVal.getDefiningOp<memref::AllocOp>();
    if (!outAlloc)
      return;

    Block *block = retOp->getBlock();

    // -----------------------------------------------------------------------
    // Scan backwards from return, collecting concat-copy triples.
    // A triple is: (copyOp, subviewOp-of-outAlloc, srcAlloc-or-forResult)
    // -----------------------------------------------------------------------
    struct ConcatCopy {
      memref::CopyOp copyOp;
      memref::SubViewOp subviewOp;
      memref::AllocOp srcAlloc;  // intermediate alloc
      scf::ForOp srcFor;         // for loop whose iter_arg init is srcAlloc
    };
    SmallVector<ConcatCopy> toEliminate;

    auto it = std::prev(retOp->getIterator());
    while (true) {
      auto copyOp = dyn_cast<memref::CopyOp>(&*it);
      if (!copyOp)
        break;

      Value src = copyOp.getSource();
      Value dst = copyOp.getTarget();

      // dst must be a subview of outAlloc.
      auto subviewOp = dst.getDefiningOp<memref::SubViewOp>();
      if (!subviewOp || subviewOp.getSource() != outAlloc->getResult(0))
        break;

      // src is either (a) a memref.alloc or (b) a scf.for result whose
      // iter_arg[i] init is a memref.alloc.
      memref::AllocOp srcAlloc;
      scf::ForOp srcFor;

      if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
        srcAlloc = alloc;
      } else if (auto forOp = src.getDefiningOp<scf::ForOp>()) {
        for (auto [res, init] :
             llvm::zip(forOp.getResults(), forOp.getInitArgs())) {
          if (res == src) {
            srcAlloc = init.getDefiningOp<memref::AllocOp>();
            if (srcAlloc)
              srcFor = forOp;
            break;
          }
        }
      }
      if (!srcAlloc)
        break;

      // srcAlloc must only be used by the for (as iter_arg) or the copy.
      bool safe = true;
      for (Operation *user : srcAlloc->getResult(0).getUsers()) {
        if (srcFor && user == srcFor.getOperation())
          continue;
        if (!srcFor && user == copyOp.getOperation())
          continue;
        if (user->getBlock() == block) {
          safe = false;
          break;
        }
      }
      if (!safe)
        break;

      toEliminate.push_back({copyOp, subviewOp, srcAlloc, srcFor});

      if (it == block->begin())
        break;
      --it;
      // Step over the subview that immediately precedes this copy.
      if (&*it == subviewOp.getOperation()) {
        if (it == block->begin())
          break;
        --it;
      }
    }

    if (toEliminate.empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << "[fold-concat-alloc] eliminating "
                             << toEliminate.size() << " concat copies\n");

    // -----------------------------------------------------------------------
    // Find insertion point: before the earliest srcFor.
    // -----------------------------------------------------------------------
    Operation *insertBefore = nullptr;
    for (auto &cc : toEliminate) {
      if (cc.srcFor) {
        if (!insertBefore || cc.srcFor->isBeforeInBlock(insertBefore))
          insertBefore = cc.srcFor.getOperation();
      }
    }
    if (!insertBefore)
      insertBefore = outAlloc.getOperation(); // fallback: no for loops

    // -----------------------------------------------------------------------
    // Rebuild outAlloc + subviews before insertBefore using srcAlloc sizes.
    //
    // For each dynamic dim of outAlloc, replace memref.dim(%forResult, idx)
    // or affine.apply(... memref.dim(%forResult, idx) ...) with
    // memref.dim(%srcAlloc, idx) where %forResult aliases %srcAlloc.
    // -----------------------------------------------------------------------
    OpBuilder b(insertBefore);
    Location loc = outAlloc->getLoc();

    // Build forResult → srcAlloc mapping.
    llvm::DenseMap<Value, Value> forResMap; // forResult → srcAlloc
    for (auto &cc : toEliminate) {
      if (cc.srcFor) {
        for (auto [res, init] :
             llvm::zip(cc.srcFor.getResults(), cc.srcFor.getInitArgs())) {
          forResMap[res] = cc.srcAlloc->getResult(0);
        }
      }
    }

    // Helper: rewrite a Value (dynamic size operand) replacing for-result dims.
    auto rewriteOperand = [&](Value operand) -> Value {
      // Case 1: memref.dim(%forResult, idx) → memref.dim(%srcAlloc, idx)
      if (auto dimOp = operand.getDefiningOp<memref::DimOp>()) {
        if (forResMap.count(dimOp.getSource()))
          return b.create<memref::DimOp>(loc, forResMap[dimOp.getSource()],
                                         dimOp.getIndex());
      }
      // Case 2: affine.apply(map, [..., memref.dim(%forResult, idx), ...])
      if (auto applyOp = operand.getDefiningOp<affine::AffineApplyOp>()) {
        SmallVector<Value> newArgs;
        bool changed = false;
        for (Value arg : applyOp.getOperands()) {
          if (auto dimOp = arg.getDefiningOp<memref::DimOp>()) {
            if (forResMap.count(dimOp.getSource())) {
              newArgs.push_back(b.create<memref::DimOp>(
                  loc, forResMap[dimOp.getSource()], dimOp.getIndex()));
              changed = true;
              continue;
            }
          }
          newArgs.push_back(arg);
        }
        if (changed)
          return b.create<affine::AffineApplyOp>(loc, applyOp.getAffineMap(),
                                                  newArgs);
      }
      return operand; // already loop-independent
    };

    // Rebuild outAlloc.
    SmallVector<Value> newSizes;
    for (Value s : outAlloc.getDynamicSizes())
      newSizes.push_back(rewriteOperand(s));

    auto newOutAlloc = b.create<memref::AllocOp>(
        loc, cast<MemRefType>(outAlloc->getResultTypes()[0]), newSizes,
        outAlloc->getAttrOfType<IntegerAttr>("alignment"));

    // Rebuild subviews (their offsets/sizes may also reference for-results).
    for (auto &cc : toEliminate) {
      auto oldSV = cc.subviewOp;
      // Rewrite mixed offsets, sizes, strides.
      auto rewriteMixed = [&](SmallVector<OpFoldResult> mixed) {
        for (auto &v : mixed) {
          if (auto val = dyn_cast<Value>(v))
            v = rewriteOperand(val);
        }
        return mixed;
      };
      auto newOffsets = rewriteMixed(oldSV.getMixedOffsets());
      auto newSizesVec = rewriteMixed(oldSV.getMixedSizes());
      auto newStrides = rewriteMixed(oldSV.getMixedStrides());

      auto newSV = b.create<memref::SubViewOp>(loc, newOutAlloc.getResult(),
                                               newOffsets, newSizesVec,
                                               newStrides);
      // Replace old subview with new one.
      oldSV->getResult(0).replaceAllUsesWith(newSV.getResult());
      cc.subviewOp = newSV;
    }

    // Replace old outAlloc uses (the return value and any dim ops after loops).
    outAlloc->getResult(0).replaceAllUsesWith(newOutAlloc.getResult());

    // -----------------------------------------------------------------------
    // Redirect each for loop's iter_arg init to the new subview.
    // -----------------------------------------------------------------------
    for (auto &cc : toEliminate) {
      Value srcAllocVal = cc.srcAlloc->getResult(0);
      Value subviewVal = cc.subviewOp->getResult(0);

      if (cc.srcFor) {
        // Find the iter_arg index corresponding to srcAlloc.
        for (unsigned i = 0, e = cc.srcFor.getNumRegionIterArgs(); i < e;
             ++i) {
          if (cc.srcFor.getInitArgs()[i] != srcAllocVal)
            continue;

          // Replace uses of the region iter_arg inside the for body.
          // Types may differ (subview is strided, iter_arg is contiguous).
          // Since memref writes are in-place, we don't need the iter_arg
          // to carry the value out — just replace the body arg with subviewVal
          // directly (using a subview of subviewVal if types differ).
          BlockArgument regionArg = cc.srcFor.getRegionIterArg(i);

          // Replace region arg uses with subviewVal.
          // If types differ, replace the region block arg type to match
          // subviewVal's type, then replace uses.
          if (regionArg.getType() != subviewVal.getType()) {
            // Update the block arg type in place.
            regionArg.setType(subviewVal.getType());
            // Also update the for op's result type.
            cc.srcFor.getResult(i).setType(subviewVal.getType());
            // And update the init args type (the for op operand already
            // points to subviewVal below, which has the right type — but
            // we need to set it first).
          }
          regionArg.replaceAllUsesWith(subviewVal);

          // Point the iter_arg init to subviewVal (types now match).
          cc.srcFor.getInitArgsMutable()[i].set(subviewVal);
          // The for result now also points to subviewVal; clear its uses.
          cc.srcFor.getResult(i).replaceAllUsesWith(subviewVal);
          break;
        }
      } else {
        srcAllocVal.replaceAllUsesWith(subviewVal);
      }

      cc.copyOp.erase();
      if (srcAllocVal.use_empty())
        cc.srcAlloc.erase();
    }

    // Erase the original outAlloc (now replaced by newOutAlloc everywhere).
    // The dim ops and affine.apply that were its operands become dead and will
    // be cleaned up by canonicalize/DCE.
    if (outAlloc->getResult(0).use_empty())
      outAlloc.erase();
  }
};

std::unique_ptr<Pass> createAscendCFoldConcatAllocPass() {
  return std::make_unique<AscendCFoldConcatAllocPass>();
}

} // namespace mlir::afir
