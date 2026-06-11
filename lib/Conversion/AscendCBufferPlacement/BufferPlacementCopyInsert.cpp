//===- BufferPlacementCopyInsert.cpp - data-copy insertion ----------------===//
//
// The data-copy-insertion half of AscendCBufferPlacement, split out of
// AscendCBufferPlacementPass.cpp.  Given the inferred TPositions + per-loop
// prologue/epilogue annotations, materializes the GM<->local DataCopy/mmad/
// fixpipe ops.  Shared types + boundary functions are in BufferPlacementInternal.h.
//
//===----------------------------------------------------------------------===//

#include "BufferPlacementInternal.h"
#include "Dialect/AFIR/AFIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#include "ascir/Dialect/Asc/IR/Asc.h"

#define DEBUG_TYPE "ascendc-buffer-placement"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir::afir::buffer_placement {

/// Trace through subview/alloc chain to find the ultimate block argument (func arg).
Value traceToBlockArgument(Value value) {
  const int maxDepth = 10;
  int depth = 0;
  while (depth < maxDepth && value) {
    if (isa<BlockArgument>(value))
      return value;
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      break;
    if (auto sv = dyn_cast<memref::SubViewOp>(defOp)) {
      value = sv.getSource();
      depth++;
      continue;
    }
    break;
  }
  return Value();
}

/// Trace through subview chains to find the ultimate alloc (not block arg).
/// Returns the alloc Value if found, else empty Value.
Value traceToAlloc(Value value) {
  const int maxDepth = 10;
  int depth = 0;
  while (depth < maxDepth && value) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return Value(); // block argument, not an alloc
    if (isa<memref::AllocOp>(defOp))
      return value;
    if (auto sv = dyn_cast<memref::SubViewOp>(defOp)) {
      value = sv.getSource();
      depth++;
      continue;
    }
    return Value();
  }
  return Value();
}

/// Find the tile subview in forOp's immediate body that corresponds to the given role.
///
/// Strategy: scan ops in forOp's body (non-recursive) for subviews whose
/// ultimate source (traced through subview chains) is a func argument.
/// - "lhs"   : subview tracing to the first input of the deepest linalg.matmul
/// - "rhs"   : subview tracing to the second input of the deepest linalg.matmul
/// - "bias"  : subview tracing to a func arg that is NOT lhs/rhs (bias arg)
/// - "acc"   : alloc that feeds matmul outs (CO1 accumulator)
/// - "result": subview tracing to the output (last) func arg
///
/// For non-GM sources (srcPos != 0), finds subviews whose source traces to an
/// alloc with the matching memory_space (e.g., A1=1 for lhs, B1=3 for rhs).
///
/// Returns the subview Value (strided memref with tile shape), or empty Value.
Value findRoleSubview(StringRef role, scf::ForOp forOp,
                      uint8_t srcPos = 0) {
  func::FuncOp funcOp = forOp->getParentOfType<func::FuncOp>();
  Block &body = forOp.getRegion().front();

  if (role == "lhs" || role == "rhs") {
    if (srcPos != 0) {
      // Non-GM source: find the subview in forOp body whose source traces to
      // an alloc with memory_space == srcPos (e.g., A1=1, B1=3).
      uint8_t targetMemSpace = srcPos;
      Value found;
      for (Operation &op : body) {
        if (found) break;
        auto sv = dyn_cast<memref::SubViewOp>(&op);
        if (!sv) continue;
        Value allocSrc = traceToAlloc(sv.getResult());
        if (!allocSrc) continue;
        auto memSpace = cast<MemRefType>(allocSrc.getType()).getMemorySpace();
        if (!memSpace) continue;
        if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
          if (static_cast<uint8_t>(intAttr.getInt()) == targetMemSpace)
            found = sv.getResult();
        }
      }
      return found;
    }
    // GM source: scan the IMMEDIATE body of forOp for subviews whose ultimate source
    // traces back to arg0/arg1 (func arg = lhs/rhs).
    unsigned targetArgIdx = (role == "lhs") ? 0 : 1;
    Value found;
    for (Operation &op : body) {
      if (found) break;
      auto sv = dyn_cast<memref::SubViewOp>(&op);
      if (!sv) continue;
      Value src = traceToBlockArgument(sv.getResult());
      if (!src) continue;
      if (auto ba = dyn_cast<BlockArgument>(src)) {
        if (isa<func::FuncOp>(ba.getOwner()->getParent()->getParentOp()) &&
            ba.getArgNumber() == targetArgIdx) {
          found = sv.getResult();
        }
      }
    }
    return found;
  }

  if (role == "bias") {
    // bias subview: scan the IMMEDIATE body of forOp for subviews of func args
    // that are not the lhs/rhs func args (i.e., bias/input C arg).
    // lhs = arg0, rhs = arg1, bias = arg2 (heuristic: 3rd func memref arg).
    Value found;
    for (Operation &op : body) {
      if (found) break;
      auto sv = dyn_cast<memref::SubViewOp>(&op);
      if (!sv) continue;
      Value blockArg = traceToBlockArgument(sv.getResult());
      if (!blockArg || !isa<BlockArgument>(blockArg)) continue;
      auto ba = cast<BlockArgument>(blockArg);
      if (!isa<func::FuncOp>(ba.getOwner()->getParent()->getParentOp()))
        continue;
      // Skip non-memref args (tile size i64 params)
      if (!isa<MemRefType>(ba.getType())) continue;
      unsigned argIdx = ba.getArgNumber();
      // Skip lhs (arg0) and rhs (arg1)
      if (argIdx == 0 || argIdx == 1) continue;
      // Skip the result/output arg: it is the last memref argument.
      // Find the last memref arg index.
      unsigned lastMemrefArgIdx = 0;
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        if (isa<MemRefType>(funcOp.getArgument(i).getType()))
          lastMemrefArgIdx = i;
      }
      if (argIdx == lastMemrefArgIdx) continue;
      found = sv.getResult();
    }
    return found;
  }

  if (role == "result") {
    // result subview: subview of the output func arg (C), in forOp body.
    // It appears as the outs operand of linalg.elementwise max_signed (VECOUT→GM src)
    // or as a subview of %arg3.
    // Find a subview in the immediate body that traces to the last func arg.
    Value found;
    for (Operation &op : body) {
      if (auto sv = dyn_cast<memref::SubViewOp>(&op)) {
        Value blockArg = traceToBlockArgument(sv.getResult());
        if (blockArg && isa<BlockArgument>(blockArg)) {
          auto ba = cast<BlockArgument>(blockArg);
          if (isa<func::FuncOp>(ba.getOwner()->getParent()->getParentOp())) {
            // Pick the subview of the result arg (last memref arg used for output)
            // Heuristic: the output subview is used as outs in max_signed
            bool usedAsOutput = false;
            for (Operation *user : sv.getResult().getUsers()) {
              if (auto elemOp = dyn_cast<linalg::ElementwiseOp>(user)) {
                for (Value init : elemOp.getDpsInits()) {
                  // init may be a subview of sv.getResult()
                  if (init == sv.getResult() ||
                      traceToBlockArgument(init) == blockArg) {
                    usedAsOutput = true;
                    break;
                  }
                }
              }
              // Also: subview directly used as copy dst in existing code
              if (isa<memref::CopyOp>(user))
                usedAsOutput = true;
            }
            if (usedAsOutput) {
              found = sv.getResult();
              break;
            }
          }
        }
      }
    }
    // Fallback: return last func arg directly if no subview found
    if (!found && funcOp.getNumArguments() > 0)
      found = funcOp.getArgument(funcOp.getNumArguments() - 1);
    return found;
  }

  if (role == "src") {
    // Generic source role: find the first ins-subview of a Vector generic
    // (linalg.generic with ascendc.unit = "AiCore.Vector") inside forOp.
    Value found;
    forOp.walk([&](linalg::GenericOp genericOp) {
      if (found)
        return WalkResult::interrupt();
      auto unitAttr = genericOp->getAttrOfType<StringAttr>("ascendc.unit");
      if (!unitAttr || unitAttr.getValue() != "AiCore.Vector")
        return WalkResult::advance();
      for (Value inp : genericOp.getInputs()) {
        Operation *defOp = inp.getDefiningOp();
        if (defOp && defOp->getBlock() == &body &&
            isa<memref::SubViewOp>(defOp)) {
          found = inp;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return found;
  }

  if (role == "dst") {
    // Generic destination role: find the outs-subview of a Vector generic
    // (linalg.generic with ascendc.unit = "AiCore.Vector") inside forOp.
    Value found;
    forOp.walk([&](linalg::GenericOp genericOp) {
      if (found)
        return WalkResult::interrupt();
      auto unitAttr = genericOp->getAttrOfType<StringAttr>("ascendc.unit");
      if (!unitAttr || unitAttr.getValue() != "AiCore.Vector")
        return WalkResult::advance();
      for (Value out : genericOp.getOutputs()) {
        Operation *defOp = out.getDefiningOp();
        if (defOp && isa<memref::SubViewOp>(defOp)) {
          found = out;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return found;
  }

  return Value();
}

/// Create a memref.alloc with TPosition memory_space.
/// dynamicSizes provides runtime size values for dynamic dimensions (ShapedType::kDynamic).
Value createLocalMemref(OpBuilder &builder, Location loc,
                        ArrayRef<int64_t> shape,
                        Type elementType,
                        uint8_t tposValue,
                        ValueRange dynamicSizes = {}) {
  auto memSpace = createMemorySpace(builder.getContext(), tposValue);
  auto memrefType = MemRefType::get(shape, elementType, MemRefLayoutAttrInterface{}, memSpace);
  return builder.create<memref::AllocOp>(loc, memrefType, dynamicSizes).getResult();
}

/// Create a memref.alloc matching srcMemref's logical shape (strided layout stripped),
/// with the given TPosition memory_space.
/// For dynamic dimensions, inserts memref.dim ops to get runtime sizes.
Value createLocalMemrefLike(OpBuilder &builder, Location loc,
                            Value srcMemref,
                            uint8_t tposValue) {
  auto srcType = cast<MemRefType>(srcMemref.getType());
  // Use the logical shape (rank dims), stripping any strided layout.
  SmallVector<Value> dynSizes;
  for (auto [idx, dim] : llvm::enumerate(srcType.getShape())) {
    if (dim == ShapedType::kDynamic) {
      dynSizes.push_back(builder.create<memref::DimOp>(loc, srcMemref, idx));
    }
  }
  return createLocalMemref(builder, loc, srcType.getShape(),
                           srcType.getElementType(), tposValue, dynSizes);
}

/// Insert ascendc.data_copy_l2 operation for GM <-> L1/VECIN transfers.
void insertDataCopyL2(OpBuilder &builder, Location loc,
                      Value dst, Value src) {
  // Phase 1 使用 memref.copy 作为占位。
  // dst 和 src 的 memory_space 携带了 TPosition 信息。
  // Phase 2 在做 memref → ascendc tensor 类型转换时，
  // 将 memref.copy 替换为对应的 DataCopyL2Op / DataCopyL0Op。
  builder.create<memref::CopyOp>(loc, src, dst);
  // 注意：memref.CopyOp 的参数顺序是 (src, dst)，与直觉相反
}

/// Insert ascendc.data_copy_l0 operation for L1 <-> L0 transfers.
void insertDataCopyL0(OpBuilder &builder, Location loc,
                       Value dst, Value src) {
  // // Create DataCopyParams for L0 transfers
  // auto paramsType = ::mlir::ascendc::DataCopyParamsType::get(builder.getContext());
  // Value params = builder.create<ascendc::ConstructOp>(loc, paramsType);
  //
  // builder.create<ascendc::DataCopyL0Op>(loc, dst, src, params);

  // Phase 1 使用 memref.copy 作为占位。
  // dst 和 src 的 memory_space 携带了 TPosition 信息。
  // Phase 2 在做 memref → ascendc tensor 类型转换时，
  // 将 memref.copy 替换为对应的 DataCopyL2Op / DataCopyL0Op。
  builder.create<memref::CopyOp>(loc, src, dst);
  // 注意：memref.CopyOp 的参数顺序是 (src, dst)，与直觉相反
}

/// Insert ascendc.fixpipe operation for CO1 -> VECIN transfer.
void insertFixpipe(OpBuilder &builder, Location loc,
                   Value dst, Value src) {
  // auto elemType = cast<MemRefType>(src.getType()).getElementType();
  //
  // // Create FixpipeParams using ConstructOp
  // auto paramsType = ::mlir::ascendc::FixpipeParamsType::get(builder.getContext(), elemType);
  // Value params = builder.create<ascendc::ConstructOp>(loc, paramsType);
  //
  // builder.create<ascendc::FixpipeOp>(loc, dst, src, /*workspace=*/Value(), params);

  // Phase 1 使用 memref.copy 作为占位。
  // dst 和 src 的 memory_space 携带了 TPosition 信息。
  // Phase 2 在做 memref → ascendc tensor 类型转换时，
  // 将 memref.copy 替换为对应的 DataCopyL2Op / DataCopyL0Op。
  builder.create<memref::CopyOp>(loc, src, dst);
  // 注意：memref.CopyOp 的参数顺序是 (src, dst)，与直觉相反
}

/// Find the first memref::AllocOp in a block whose result has the given
/// memory_space integer value. Used after updateAllocMemorySpace runs.
Value findAllocByMemSpace(Block &block, uint8_t targetTPos) {
  for (auto &op : block) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
      auto memSpace = cast<MemRefType>(allocOp.getType()).getMemorySpace();
      if (!memSpace)
        continue;
      // TPositionAttr is stored as an IntegerAttr with value == tpos
      if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
        if (static_cast<uint8_t>(intAttr.getInt()) == targetTPos)
          return allocOp.getResult();
      }
    }
  }
  return Value();
}

/// Find a buffer by TPosition in posMap within a specific block scope.
/// Falls back to searching by memory_space on the alloc op itself.
Value findBufferByPosInScope(Block &block, const BufferPosMap &posMap,
                             uint8_t targetPos) {
  // First try posMap (pre-update)
  for (auto &op : block) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
      auto it = posMap.find(allocOp.getResult());
      if (it != posMap.end() && it->second == targetPos) {
        return allocOp.getResult();
      }
    }
  }
  // Fallback: search by memory_space attr on alloc (post-update)
  return findAllocByMemSpace(block, targetPos);
}

/// Find the source buffer for a given TPosition by searching enclosing scopes.
/// For non-GM sources (A1, B1, etc.), we look for allocs with that memory_space
/// in the enclosing blocks.
Value findSourceBufferByPos(uint8_t srcPos, scf::ForOp forOp) {
  // Walk up the parent hierarchy to find an alloc with the matching memory_space
  Operation *parent = forOp->getParentOp();
  while (parent) {
    for (Region &region : parent->getRegions()) {
      for (Block &block : region) {
        for (auto &op : block) {
          if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
            auto memSpace = cast<MemRefType>(allocOp.getType()).getMemorySpace();
            if (!memSpace)
              continue;
            if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
              if (static_cast<uint8_t>(intAttr.getInt()) == srcPos)
                return allocOp.getResult();
            }
          }
        }
      }
    }
    if (isa<func::FuncOp>(parent))
      break;
    parent = parent->getParentOp();
  }
  return Value();
}

/// Insert data copy operations based on loop annotations.
void insertCopiesForLoop(scf::ForOp forOp,
                         const LoopAnnotation &ann,
                         const BufferPosMap &posMap,
                         OpBuilder &builder) {
  Block &body = forOp.getRegion().front();

  // ── Prologue ──────────────────────────────────────────────────────────────
  // For each prologue task, find the source tile subview (or L1 alloc), then
  // insert the alloc + copy immediately AFTER the source value is defined.
  // This ensures the dim ops used to compute alloc sizes dominate their uses.
  //
  // When the source is a block arg (not defined in forOp body), we use a
  // "watermark" op to track where the last prologue op was inserted, ensuring
  // sequential ordering of prologue ops.

  Operation *prologueWatermark = nullptr; // last inserted op in this prologue sequence
  // Track new local buffers created during prologue: role -> dstMemref.
  // Used to redirect matmul ins and bias operands after all prologues are inserted.
  DenseMap<StringRef, Value> prologueBufs;
  // Track A2/B2 buffers that need dealloc at end of loop body.
  SmallVector<Value> prologueDeallocBufs;

  for (const auto &task : ann.prologue) {
    Value srcMemref;

    // Always try findRoleSubview first: it finds the tile-sized subview in
    // forOp's immediate body, giving the correct (tile) shape for the alloc.
    // For GM->A1: finds subview of %arg0/%arg1 with tile shape [TB_M x K].
    // For A1->A2: finds subview of A1 alloc with K-slice shape [tb_M x t_K].
    srcMemref = findRoleSubview(task.role, forOp, task.srcPos);

    if (!srcMemref) {
      if (task.srcPos == 0) {
        // GM source: no subview in body, fall back to the func arg directly.
        // This handles test-only scenarios where the loop body has no subviews.
        // Map role to func arg: lhs=arg0, rhs=arg1, bias=arg2.
        auto funcOp = forOp->getParentOfType<func::FuncOp>();
        if (task.role == "lhs" && funcOp && funcOp.getNumArguments() > 0)
          srcMemref = funcOp.getArgument(0);
        else if (task.role == "rhs" && funcOp && funcOp.getNumArguments() > 1)
          srcMemref = funcOp.getArgument(1);
        else if (task.role == "bias" && funcOp && funcOp.getNumArguments() > 2)
          srcMemref = funcOp.getArgument(2);
      } else if (task.role == "lhs" || task.role == "rhs") {
        // Non-GM source (A1->A2 or B1->B2):
        // Find the L1 alloc (A1/B1) and the GM subview in forOp body that has
        // the correct tile shape (offsets/sizes for the K-slice).
        // Create a subview of the L1 alloc using those parameters.
        Value l1Alloc = findSourceBufferByPos(task.srcPos, forOp);
        if (l1Alloc) {
          // Find the GM subview in forOp body (traces to arg0/arg1) to get the
          // tile offsets/sizes/strides for the K-slice.
          unsigned targetArgIdx = (task.role == "lhs") ? 0 : 1;
          memref::SubViewOp gmSubview;
          for (Operation &op : body) {
            auto sv = dyn_cast<memref::SubViewOp>(&op);
            if (!sv) continue;
            Value src = traceToBlockArgument(sv.getResult());
            if (!src) continue;
            if (auto ba = dyn_cast<BlockArgument>(src)) {
              if (isa<func::FuncOp>(ba.getOwner()->getParent()->getParentOp()) &&
                  ba.getArgNumber() == targetArgIdx) {
                gmSubview = sv;
                break;
              }
            }
          }
          if (gmSubview) {
            // Insert a subview of l1Alloc using the GM subview's parameters.
            // The offsets/sizes/strides encode the K-slice within the L1 buffer.
            Operation *defOp = gmSubview.getResult().getDefiningOp();
            if (defOp && defOp->getBlock() == &body)
              builder.setInsertionPointAfter(defOp);
            else
              builder.setInsertionPointToStart(&body);
            auto l1Type = cast<MemRefType>(l1Alloc.getType());
            auto inferredType = memref::SubViewOp::inferRankReducedResultType(
                gmSubview.getType().getShape(),
                l1Type,
                gmSubview.getStaticOffsets(),
                gmSubview.getStaticSizes(),
                gmSubview.getStaticStrides());
            srcMemref = builder.create<memref::SubViewOp>(
                forOp.getLoc(),
                inferredType,
                l1Alloc,
                gmSubview.getMixedOffsets(),
                gmSubview.getMixedSizes(),
                gmSubview.getMixedStrides()).getResult();
          }
        }
      } else {
        // Other non-GM fallback.
        srcMemref = findSourceBufferByPos(task.srcPos, forOp);
      }
    }

    if (!srcMemref)
      continue;

    // Set insertion point:
    // - If src is defined in this body, insert after its defining op.
    // - Otherwise, insert after the last prologue op (watermark), or at body start.
    Operation *defOp = srcMemref.getDefiningOp();
    if (defOp && defOp->getBlock() == &body) {
      builder.setInsertionPointAfter(defOp);
      prologueWatermark = nullptr; // reset: next body-start case starts after defOp chain
    } else if (prologueWatermark) {
      builder.setInsertionPointAfter(prologueWatermark);
    } else {
      builder.setInsertionPointToStart(&body);
    }

    // Create local alloc matching the source's logical shape (strided layout stripped).
    Value dstMemref = createLocalMemrefLike(builder, forOp.getLoc(),
                                            srcMemref, task.dstPos);
    prologueBufs[task.role] = dstMemref;

    // Insert appropriate copy operation based on source position
    if (task.srcPos == 0) { // GM -> L1/VECIN
      insertDataCopyL2(builder, forOp.getLoc(), dstMemref, srcMemref);
    } else if (task.srcPos == 1 || task.srcPos == 3) { // A1/B1 -> A2/B2
      insertDataCopyL0(builder, forOp.getLoc(), dstMemref, srcMemref);
    }
    // All prologue buffers need dealloc at end of this loop body.
    prologueDeallocBufs.push_back(dstMemref);
    // Track the last inserted op (the copy) as the new watermark.
    // The copy is inserted by builder at the current position, so it's the
    // op just before the current insertion point.
    if (!defOp || defOp->getBlock() != &body) {
      // Get the op just before the current insertion point (the copy we just inserted)
      Block::iterator insertPt = builder.getInsertionPoint();
      if (insertPt != body.begin()) {
        prologueWatermark = &*std::prev(insertPt);
      }
    }
  }

  // ── Post-prologue operand redirections ────────────────────────────────────
  //
  // After inserting A2/B2 allocs, redirect the matmul's ins operands from
  // the original subviews to the new A2/B2 buffers (Problem 1).
  //
  // The matmul is inside a nested loop (for_K) within forOp's body.
  // We walk forOp's body to find the first matmul and reassign its inputs.
  if (prologueBufs.count("lhs") || prologueBufs.count("rhs")) {
    forOp.walk([&](linalg::MatmulOp matmulOp) {
      if (auto lhsBuf = prologueBufs.lookup("lhs"))
        matmulOp.getInputsMutable()[0].assign(lhsBuf);
      if (auto rhsBuf = prologueBufs.lookup("rhs"))
        matmulOp.getInputsMutable()[1].assign(rhsBuf);
    });
  }

  // Insert deallocs for short-lived prologue buffers (A2/B2) at end of body.
  if (!prologueDeallocBufs.empty()) {
    builder.setInsertionPoint(body.getTerminator());
    for (Value buf : prologueDeallocBufs)
      builder.create<memref::DeallocOp>(forOp.getLoc(), buf);
  }

  // Redirect bias operands of all linalg.elementwise ops inside forOp
  // to use the VECIN buffer (inserted by prologue bias:GM->VECIN task).
  // Any elementwise op whose ins[i] traces to a GM memref (no memory_space)
  // and ultimately to the same func arg as the bias VECIN buffer's source
  // is redirected — regardless of elementwise kind (add, mul, sub, etc.).
  //
  // If the elementwise op lives in an inner loop and operates on a sub-tile
  // smaller than the VECIN buffer, we create a subview of the VECIN buffer
  // matching the outs operand shape/offsets so sizes are consistent.
  if (auto vecinBias = prologueBufs.lookup("bias")) {
    forOp.walk([&](linalg::ElementwiseOp elemOp) {
      for (auto &inputOperand : elemOp.getInputsMutable()) {
        Value inp = inputOperand.get();
        auto memTy = dyn_cast<MemRefType>(inp.getType());
        // Only redirect GM operands (no memory_space).
        if (!memTy || memTy.getMemorySpace())
          continue;
        // Check that this operand traces to a func arg that is neither
        // lhs (arg0) nor rhs (arg1) — i.e., it is a bias-like input.
        Value ultimateSrc = traceToBlockArgument(inp);
        if (!ultimateSrc)
          continue;
        auto ba = dyn_cast<BlockArgument>(ultimateSrc);
        if (!ba)
          continue;
        if (!isa<func::FuncOp>(ba.getOwner()->getParent()->getParentOp()))
          continue;
        unsigned argIdx = ba.getArgNumber();
        if (argIdx == 0 || argIdx == 1)
          continue; // skip lhs/rhs args

        // Determine the replacement: either vecinBias directly (if shapes
        // match) or a subview of vecinBias matching the outs operand's tile.
        Value replacement = vecinBias;
        auto vecinTy = cast<MemRefType>(vecinBias.getType());

        // Check the outs operand to see if the elementwise operates on a
        // sub-tile of vecinBias. Use the outs subview offsets/sizes.
        if (!elemOp.getOutputs().empty()) {
          Value outsVal = elemOp.getOutputs().front();
          auto outsTy = dyn_cast<MemRefType>(outsVal.getType());
          // If outs has a different static rank-0 shape it is a subview; get
          // the offsets/sizes from the original GM bias subview in the body.
          // More reliably: if vecinBias shape != outs shape, create subview.
          // We derive offsets from the original GM bias subview that inp came
          // from — it carries the correct loop-relative offsets.
          if (outsTy && outsTy.getRank() == vecinTy.getRank()) {
            // Collect mixed sizes from outs (for dynamic shapes).
            auto outsSubview = dyn_cast_or_null<memref::SubViewOp>(
                outsVal.getDefiningOp());
            // Also try the inp subview for offset information.
            auto inpSubview = dyn_cast_or_null<memref::SubViewOp>(
                inp.getDefiningOp());

            // Use the inp (GM bias subview) offsets/sizes to slice vecinBias.
            // The GM subview carries the correct tile offsets (e.g., [%arg11,
            // %arg12] with sizes [%8, %9]).
            if (inpSubview) {
              OpBuilder svBuilder(elemOp);
              // Build offsets/sizes/strides from the GM subview but apply to
              // vecinBias (which has VECIN memory_space).
              auto inferredType =
                  cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
                      outsTy.getShape(), vecinTy,
                      inpSubview.getStaticOffsets(),
                      inpSubview.getStaticSizes(),
                      inpSubview.getStaticStrides()));
              replacement = svBuilder
                                .create<memref::SubViewOp>(
                                    elemOp.getLoc(), inferredType, vecinBias,
                                    inpSubview.getMixedOffsets(),
                                    inpSubview.getMixedSizes(),
                                    inpSubview.getMixedStrides())
                                .getResult();
            } else if (outsSubview) {
              // Fallback: use outs subview offsets (handles non-subview bias).
              OpBuilder svBuilder(elemOp);
              auto inferredType =
                  cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
                      outsTy.getShape(), vecinTy,
                      outsSubview.getStaticOffsets(),
                      outsSubview.getStaticSizes(),
                      outsSubview.getStaticStrides()));
              replacement = svBuilder
                                .create<memref::SubViewOp>(
                                    elemOp.getLoc(), inferredType, vecinBias,
                                    outsSubview.getMixedOffsets(),
                                    outsSubview.getMixedSizes(),
                                    outsSubview.getMixedStrides())
                                .getResult();
            }
          }
        }
        inputOperand.assign(replacement);
      }
    });
  }

  // Redirect linalg.generic ins operands to the VECIN buffer created for
  // the "src" prologue role. The srcMemref used to create the VECIN alloc
  // was the first ins-subview of the Vector generic (from findRoleSubview).
  // We only redirect that specific operand (matched by Value identity).
  if (auto vecinSrc = prologueBufs.lookup("src")) {
    // The prologue loop recorded srcMemref via findRoleSubview("src"), which
    // returns the first ins-subview of the Vector generic. We need to find
    // which Value was used as srcMemref for the "src" task.
    // Since prologueBufs["src"] = dstMemref (the new VECIN alloc), and
    // srcMemref is the original GM subview, we can recover it by walking
    // the generic and matching by shape/element-type with the VECIN alloc.
    forOp.walk([&](linalg::GenericOp genericOp) {
      auto unitAttr = genericOp->getAttrOfType<StringAttr>("ascendc.unit");
      if (!unitAttr || unitAttr.getValue() != "AiCore.Vector")
        return;
      auto vecinTy = cast<MemRefType>(vecinSrc.getType());
      bool redirected = false;
      for (auto &inputOperand : genericOp.getInputsMutable()) {
        if (redirected)
          break;
        Value inp = inputOperand.get();
        auto memTy = dyn_cast<MemRefType>(inp.getType());
        if (!memTy)
          continue;
        // Only redirect if shape and element type match the VECIN alloc.
        if (memTy.getShape() != vecinTy.getShape() ||
            memTy.getElementType() != vecinTy.getElementType())
          continue;
        // Only redirect GM operands (no memory_space or memory_space == 0).
        if (auto ms = memTy.getMemorySpace()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(ms))
            if (intAttr.getInt() != 0)
              continue; // skip non-GM
        }
        inputOperand.assign(vecinSrc);
        redirected = true;
      }
    });
  }

  // ── Epilogue ──────────────────────────────────────────────────────────────
  //
  // CO1→VECIN (acc): CO1 alloc is defined inside for_K body; it remains
  // accessible after for_K ends (SSA value dominates the rest of its block).
  // The fixpipe copy and VECIN alloc are inserted AFTER forOp so that they
  // run once after all K iterations complete.
  //
  // VECOUT→GM (result): VECOUT alloc is defined in an inner loop body and
  // does NOT dominate positions after for_TB_N. The copy must be inserted
  // INSIDE the loop body (before the terminator), where both VECOUT and the
  // result subview are in scope.

  for (const auto &task : ann.epilogue) {
    if (task.srcPos == 7 && task.dstPos == 9) { // CO1 -> VECIN: Fixpipe
      // CO1 buffer is allocated inside forOp body.
      Value co1Buf = findBufferByPosInScope(body, posMap, 7);
      if (!co1Buf)
        co1Buf = findSourceBufferByPos(7, forOp);
      if (!co1Buf)
        continue;

      // First collect dynamic sizes from co1Buf (before any replacement).
      // Then insert VECIN alloc + copy after forOp.
      builder.setInsertionPointAfter(forOp);
      Value vecinMemref = createLocalMemrefLike(builder, forOp.getLoc(),
                                                co1Buf, task.dstPos);
      insertFixpipe(builder, forOp.getLoc(), vecinMemref, co1Buf);

      // Redirect uses of CO1 that appear AFTER forOp (e.g., add's ins[0])
      // to the new VECIN buffer.
      // Exclude dim ops that were just inserted (they use co1Buf to compute sizes).
      co1Buf.replaceUsesWithIf(vecinMemref, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        // Only replace uses in the same block, strictly after forOp.
        if (user->getBlock() != forOp->getBlock())
          return false;
        if (user->isBeforeInBlock(forOp) || user == forOp)
          return false;
        // Skip the dim ops that createLocalMemrefLike inserted for vecinMemref.
        if (isa<memref::DimOp>(user))
          return false;
        // Skip the fixpipe copy itself.
        if (isa<memref::CopyOp>(user))
          return false;
        // Skip dealloc: CO1 must still be freed by its own dealloc.
        if (isa<memref::DeallocOp>(user))
          return false;
        return true;
      });

      // Insert dealloc for the VECIN buffer at the end of forOp's parent block.
      // The VECIN buffer is consumed by the elementwise op(s) after forOp;
      // dealloc after the last use (parent block terminator is a safe position).
      builder.setInsertionPoint(forOp->getBlock()->getTerminator());
      builder.create<memref::DeallocOp>(forOp.getLoc(), vecinMemref);

    } else if (task.dstPos == 0) { // VECOUT -> GM: DataCopy
      // For vector-only ops (no pre-existing VECOUT alloc), we must:
      //   1. Find the linalg.generic outs operand (a GM subview).
      //   2. Create a new VECOUT alloc matching its shape.
      //   3. Redirect the generic outs to the VECOUT alloc.
      //   4. Insert memref.copy(VECOUT, GM_subview) before terminator.
      //   5. Insert memref.dealloc(VECOUT) before terminator.
      //
      // If a VECOUT alloc already exists in posMap, the existing copy from
      // updateAllocMemorySpace handles it — no further action needed.
      bool existingVecout = false;
      for (auto &[allocVal, pos] : posMap) {
        if (pos == 10) { // VECOUT
          existingVecout = true;
          break;
        }
      }
      if (existingVecout)
        continue;

      // Find the LAST Vector generic inside forOp (the one that writes to GM).
      // When multiple vector generics are chained (e.g. add → leaky_relu),
      // only the final one should be redirected to VECOUT; intermediate results
      // use VECCALC buffers that are already allocated by Rule B.
      linalg::GenericOp vecGeneric;
      forOp.walk([&](linalg::GenericOp genericOp) {
        auto unitAttr = genericOp->getAttrOfType<StringAttr>("ascendc.unit");
        if (unitAttr && unitAttr.getValue() == "AiCore.Vector")
          vecGeneric = genericOp; // keep updating to get the last one
        return WalkResult::advance();
      });
      if (!vecGeneric)
        continue;

      if (vecGeneric.getNumDpsInits() == 0)
        continue;
      Value gmSubview = vecGeneric.getDpsInitOperand(0)->get();

      // Create VECOUT alloc matching the GM subview's logical shape.
      builder.setInsertionPoint(vecGeneric);
      Value vecoutAlloc =
          createLocalMemrefLike(builder, forOp.getLoc(), gmSubview, 10);

      // Redirect generic outs from GM subview to VECOUT alloc.
      vecGeneric.getDpsInitsMutable()[0].assign(vecoutAlloc);

      // Insert VECOUT->GM copy and dealloc before the terminator of the block
      // that contains vecGeneric (not forOp's body, which may be an outer
      // loop).  vecoutAlloc is defined in vecGeneric's block; using
      // body.getTerminator() would place the copy outside that scope, causing
      // a dominance violation when vecGeneric is in a nested loop.
      builder.setInsertionPoint(vecGeneric->getBlock()->getTerminator());
      insertDataCopyL2(builder, forOp.getLoc(), gmSubview, vecoutAlloc);
      builder.create<memref::DeallocOp>(forOp.getLoc(), vecoutAlloc);
    }
  }

}

/// Process all loops and insert copy operations.
void insertCopiesForLoops(func::FuncOp funcOp,
                         const DenseMap<Operation *, LoopAnnotation> &loopAnns,
                         BufferPosMap &posMap,
                         OpBuilder &builder) {
  // Process loops from outermost to innermost to ensure alloc lifetimes are correct
  SmallVector<scf::ForOp> loops;
  for (const auto &[op, ann] : loopAnns) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      loops.push_back(forOp);
    }
  }

  // Sort loops by depth (outermost first)
  llvm::sort(loops, [](scf::ForOp a, scf::ForOp b) {
    int depthA = 0, depthB = 0;
    for (Operation *p = a; p; p = p->getParentOp()) depthA++;
    for (Operation *p = b; p; p = p->getParentOp()) depthB++;
    return depthA < depthB;
  });

  // Insert copies for each loop
  for (scf::ForOp forOp : loops) {
    auto it = loopAnns.find(forOp);
    if (it != loopAnns.end()) {
      insertCopiesForLoop(forOp, it->second, posMap, builder);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "insertCopiesForLoops: Processed " << loops.size()
                 << " loop annotations\n";
  });
}

} // namespace mlir::afir::buffer_placement
