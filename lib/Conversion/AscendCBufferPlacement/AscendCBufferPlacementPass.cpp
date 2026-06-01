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

#include "Conversion/AscendCBufferPlacement/AscendCBufferPlacementPass.h"
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
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#include "ascir/Dialect/Asc/IR/Asc.h"

#define GEN_PASS_DECL_ASCENDCBUFFERPLACEMENTPASS
#define GEN_PASS_DEF_ASCENDCBUFFERPLACEMENTPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::ascendc;
using namespace mlir::afir::buffer_placement;

#define DEBUG_TYPE "ascendc-buffer-placement"

// CopyTask / LoopAnnotation / BufferPosMap: see BufferPlacementInternal.h.

//===----------------------------------------------------------------------===//
// TPosition utilities
//===----------------------------------------------------------------------===//

namespace {

/// Parse TPosition from string name to enum value.
uint8_t parseTPositionValue(StringRef name) {
  return StringSwitch<uint8_t>(name)
      .Case("GM", 0)
      .Case("A1", 1)
      .Case("A2", 2)
      .Case("B1", 3)
      .Case("B2", 4)
      .Case("CO1", 7)
      .Case("VECIN", 9)
      .Case("VECOUT", 10)
      .Case("VECCALC", 11)
      .Default(0); // Default to GM
}

} // namespace

/// Create TPosition memory space attribute.  In the buffer_placement namespace
/// (declared in BufferPlacementInternal.h) so the split-out copy-insertion file
/// can call it.
namespace mlir::afir::buffer_placement {
Attribute createMemorySpace(MLIRContext *ctx, uint8_t tposValue) {
  auto ascendcPos = static_cast<::mlir::ascendc::TPosition>(tposValue);
  return ::mlir::ascendc::TPositionAttr::get(ctx, ascendcPos);
}
} // namespace mlir::afir::buffer_placement

//===----------------------------------------------------------------------===//
// Annotation parsing
//===----------------------------------------------------------------------===//

namespace {

/// Parse a single copy task string like "lhs:GM->A1".
FailureOr<CopyTask> parseCopyTask(StringRef entry) {
  auto [role, rest] = entry.split(':');
  auto [srcStr, dstStr] = rest.split("->");

  if (role.empty() || srcStr.empty() || dstStr.empty())
    return failure();

  CopyTask task;
  task.role = role.trim();
  task.srcPos = parseTPositionValue(srcStr.trim());
  task.dstPos = parseTPositionValue(dstStr.trim());
  return task;
}

/// Parse prologue/epilogue annotation string.
/// Format: "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"
SmallVector<CopyTask> parseCopyTasks(StringRef annotation) {
  SmallVector<CopyTask> tasks;
  if (annotation.empty())
    return tasks;

  SmallVector<StringRef, 8> entries;
  annotation.split(entries, ',');
  for (auto entry : entries) {
    auto task = parseCopyTask(entry.trim());
    if (succeeded(task)) {
      tasks.push_back(*task);
    }
  }
  return tasks;
}

/// Collect all loop annotations in function.
DenseMap<Operation *, LoopAnnotation> collectLoopAnnotations(func::FuncOp funcOp) {
  DenseMap<Operation *, LoopAnnotation> annotations;

  funcOp.walk([&](scf::ForOp forOp) {
    LoopAnnotation ann;

    if (auto prologueAttr = forOp->getAttrOfType<StringAttr>("ascendc.prologue")) {
      ann.prologue = parseCopyTasks(prologueAttr.getValue());
    }

    if (auto epilogueAttr = forOp->getAttrOfType<StringAttr>("ascendc.epilogue")) {
      ann.epilogue = parseCopyTasks(epilogueAttr.getValue());
    }

    ann.isParallel = forOp->hasAttr("ascendc.parallel");

    if (!ann.prologue.empty() || !ann.epilogue.empty() || ann.isParallel) {
      annotations[forOp] = std::move(ann);
    }
  });

  return annotations;
}

} // namespace

//===----------------------------------------------------------------------===//
// TPosition inference for allocs
//===----------------------------------------------------------------------===//

namespace {

// BufferPosMap: see BufferPlacementInternal.h.

/// Helper function to trace through subview operations to find the ultimate alloc
Value findUltimateSource(Value value) {
  // Limit the depth to avoid infinite loops in case of cycles
  const int maxDepth = 10;
  int depth = 0;

  while (depth < maxDepth) {
    // If the value is a block argument or has no defining op, return it
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp)
      return value;

    // If it's an alloc operation, we've found the source
    if (isa<memref::AllocOp>(definingOp))
      return value;

    // If it's a subview operation, continue tracing to its source
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(definingOp)) {
      value = subviewOp.getSource();
      depth++;
      continue;
    }

    // If it's a cast operation, continue tracing through it
    if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
      value = castOp.getSource();
      depth++;
      continue;
    }

    // For other operations, we can't trace further
    break;
  }

  return Value(); // Return null if we couldn't find an alloc
}

/// Like findUltimateSource but also traces through scf.for iter_args.
/// Used for Rule C (matmul CO1 accumulator) where the bufferizer may thread
/// the accumulator alloc through a K-loop as an iter_arg.
Value findUltimateSourceThroughIter(Value value) {
  const int maxDepth = 15;
  int depth = 0;
  while (depth < maxDepth) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto forOp =
          dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (forOp) {
        unsigned iterArgIdx = blockArg.getArgNumber() - 1; // 0 = induction var
        auto initArgs = forOp.getInitArgs();
        if (iterArgIdx < initArgs.size()) {
          value = initArgs[iterArgIdx];
          depth++;
          continue;
        }
      }
      return value; // block arg not from iter_arg
    }
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return value;
    if (isa<memref::AllocOp>(defOp))
      return value;
    if (auto sv = dyn_cast<memref::SubViewOp>(defOp)) {
      value = sv.getSource();
      depth++;
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
      value = castOp.getSource();
      depth++;
      continue;
    }
    break;
  }
  return Value();
}

/// Infer TPosition for each memref::AllocOp based on:
/// - Rule A: Loop prologue/epilogue annotations
/// - Rule B: Vector op def-use chains (VECCALC vs VECOUT)
/// - Rule C: Matmul outs operand is always CO1
void inferBufferPositions(func::FuncOp funcOp,
                         const DenseMap<Operation *, LoopAnnotation> &loopAnns,
                         BufferPosMap &posMap) {
  // Rule C: matmul outs alloc -> CO1
  // The matmul accumulator may be threaded through a K-reduction scf.for as an
  // iter_arg (generated by the bufferizer when allow-return-allocs-from-loops
  // is true).  Use findUltimateSourceThroughIter to trace past iter_args.
  funcOp.walk([&](linalg::MatmulOp matmulOp) {
    Value out = matmulOp.getDpsInitOperand(0)->get();
    Value cubeSourceOp = findUltimateSourceThroughIter(out);
    if (!cubeSourceOp)
      return;
    if (Operation *definingOp = cubeSourceOp.getDefiningOp()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(definingOp)) {
        posMap[allocOp.getResult()] = 7; // CO1
      }
    }
  });

  // Rule B: Vector op outs alloc -> VECCALC or VECOUT
  // Handle both linalg::GenericOp and linalg::ElementwiseOp
  auto processVectorOp = [&](Operation *op) {
    auto unitAttr = op->getAttrOfType<StringAttr>("ascendc.unit");
    if (!unitAttr || unitAttr.getValue() != "AiCore.Vector")
      return;

    // Get the DPS output operand
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dpsOp || dpsOp.getNumDpsInits() == 0)
      return;

    Value out = dpsOp.getDpsInitOperand(0)->get();
    Value vectorSourceOp = findUltimateSource(out);
    Operation *definingOp = vectorSourceOp.getDefiningOp();
    if (!definingOp || !isa<memref::AllocOp>(definingOp))
      return;

    // Skip allocs that are directly returned by the function (they are GM
    // output buffers, not on-chip VECOUT/VECCALC buffers).  This can happen
    // when fold-concat-alloc redirects linalg generics to write directly into
    // the output alloc via a subview.
    for (func::ReturnOp retOp : funcOp.getOps<func::ReturnOp>()) {
      for (Value retVal : retOp.getOperands()) {
        if (retVal == vectorSourceOp)
          return;
      }
    }

    // Check if there are any subsequent Vector consumers (excluding op itself)
    bool hasVectorConsumer = false;
    for (Operation *user : out.getUsers()) {
      if (user == op)
        continue; // Skip the op itself (it uses out as output)
      auto userUnit = user->getAttrOfType<StringAttr>("ascendc.unit");
      if (userUnit && userUnit.getValue() == "AiCore.Vector") {
        hasVectorConsumer = true;
        break;
      }
    }

    posMap[vectorSourceOp] = hasVectorConsumer ? 11 : 10; // VECCALC : VECOUT
  };

  funcOp.walk([&](linalg::GenericOp genericOp) {
    processVectorOp(genericOp);
  });
  funcOp.walk([&](linalg::ElementwiseOp elemOp) {
    processVectorOp(elemOp);
  });

  // Rule A: Prologue/epilogue annotations are handled during copy insertion
  (void)loopAnns;
}

} // namespace

//===----------------------------------------------------------------------===//
// Function argument memory space annotation (GM)
//===----------------------------------------------------------------------===//

namespace {

/// Update all memref function arguments to have GM memory_space.
void annotateGMArgs(func::FuncOp funcOp, OpBuilder &builder) {
  auto *ctx = funcOp.getContext();
  auto gmSpace = createMemorySpace(ctx, 0); // GM = 0

  // Build new argument types with GM memory_space
  SmallVector<Type, 4> newArgTypes;
  for (Type argType : funcOp.getArgumentTypes()) {
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      // Create new memref type with GM memory_space
      newArgTypes.push_back(MemRefType::get(
          memrefType.getShape(), memrefType.getElementType(),
          memrefType.getLayout(), gmSpace));
    } else {
      newArgTypes.push_back(argType);
    }
  }

  // Update block argument types
  for (auto [arg, newType] :
       llvm::zip(funcOp.getArguments(), newArgTypes)) {
    arg.setType(newType);
  }

  // Update function type
  auto newFuncType =
      builder.getFunctionType(newArgTypes, funcOp.getFunctionType().getResults());
  funcOp.setFunctionType(newFuncType);
}

/// Propagate memory_space from memref source to subview results.
/// After annotating function args with GM memory_space, all subview ops
/// that derive from those args must also preserve memory_space.
/// Uses fixed-point iteration to handle subview chains correctly.
void propagateSubviewMemorySpace(func::FuncOp funcOp, RewriterBase &rewriter) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<memref::SubViewOp> toUpdate;
    funcOp.walk([&](memref::SubViewOp subviewOp) {
      Value source = subviewOp.getSource();
      auto srcType = dyn_cast<MemRefType>(source.getType());
      if (!srcType || !srcType.getMemorySpace())
        return;
      auto resType = cast<MemRefType>(subviewOp.getType());
      if (resType.getMemorySpace())
        return;
      toUpdate.push_back(subviewOp);
    });

    for (auto subviewOp : toUpdate) {
      Value source = subviewOp.getSource();
      auto sourceType = cast<MemRefType>(source.getType());
      auto resultType = cast<MemRefType>(subviewOp.getType());

      // Infer the correct result type by keeping existing layout but adding memory_space.
      // If source has flat layout (no strides) and result has strided layout, keep the
      // strided layout (it encodes the actual data layout of the subview).
      // We only add the memory_space from the source.
      auto newResultType = MemRefType::get(
          resultType.getShape(), resultType.getElementType(),
          resultType.getLayout(), sourceType.getMemorySpace());

      // Verify the new type is compatible: check that the strides of the result
      // type are consistent with a subview of source. If not, use inferred layout.
      {
        auto inferredType = memref::SubViewOp::inferRankReducedResultType(
            resultType.getShape(),
            sourceType,
            subviewOp.getStaticOffsets(),
            subviewOp.getStaticSizes(),
            subviewOp.getStaticStrides());
        // If the existing layout differs from the inferred layout, use inferred.
        if (inferredType.getLayout() != resultType.getLayout()) {
          newResultType = MemRefType::get(
              inferredType.getShape(), inferredType.getElementType(),
              inferredType.getLayout(), sourceType.getMemorySpace());
        }
      }

      rewriter.setInsertionPoint(subviewOp);
      auto newSubview = rewriter.create<memref::SubViewOp>(
          subviewOp.getLoc(),
          newResultType,
          source,
          subviewOp.getMixedOffsets(),
          subviewOp.getMixedSizes(),
          subviewOp.getMixedStrides());

      rewriter.replaceAllUsesWith(subviewOp.getResult(), newSubview.getResult());
      rewriter.eraseOp(subviewOp);
      changed = true;
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Alloc memory_space update
//===----------------------------------------------------------------------===//

// updateAllocMemorySpace lives in the buffer_placement namespace (declared in
// BufferPlacementInternal.h) so the split-out copy-insertion file can call it.
namespace mlir::afir::buffer_placement {

/// Update existing alloc ops to have correct memory_space.
/// Updates posMap in-place so keys remain valid after alloc replacement.
void updateAllocMemorySpace(BufferPosMap &posMap, RewriterBase &rewriter) {
  // Collect (old alloc, tpos) pairs first to avoid iterator invalidation.
  SmallVector<std::pair<memref::AllocOp, uint8_t>> toUpdate;
  for (auto [value, tposValue] : posMap) {
    if (auto allocOp = dyn_cast_or_null<memref::AllocOp>(value.getDefiningOp()))
      toUpdate.push_back({allocOp, tposValue});
  }

  DenseMap<Value, Value> allocRemap; // old result -> new result
  for (auto [allocOp, tposValue] : toUpdate) {
    auto oldType = cast<MemRefType>(allocOp.getType());
    auto newSpace = createMemorySpace(allocOp.getContext(), tposValue);
    auto newType = MemRefType::get(oldType.getShape(), oldType.getElementType(),
                                   oldType.getLayout(), newSpace);

    rewriter.setInsertionPoint(allocOp);
    auto newAlloc = rewriter.create<memref::AllocOp>(
        allocOp.getLoc(), newType, allocOp.getDynamicSizes());

    rewriter.replaceAllUsesWith(allocOp.getResult(), newAlloc.getResult());
    allocRemap[allocOp.getResult()] = newAlloc.getResult();
    rewriter.eraseOp(allocOp);

    // If the new alloc is used as an scf.for init arg, update the
    // corresponding iter_arg block-argument type and result type so that
    // the IR remains type-consistent.
    for (Operation *user : llvm::make_early_inc_range(newAlloc.getResult().getUsers())) {
      auto forOp = dyn_cast<scf::ForOp>(user);
      if (!forOp)
        continue;
      // Find which init-arg index(es) correspond to newAlloc.
      for (auto [initIdx, initVal] : llvm::enumerate(forOp.getInitArgs())) {
        if (initVal != newAlloc.getResult())
          continue;
        // Block arg index = initIdx + 1 (0 is the induction variable).
        BlockArgument iterArg = forOp.getBody()->getArgument(initIdx + 1);
        iterArg.setType(newType);
        // Update the for op result type.
        forOp.getResult(initIdx).setType(newType);
      }
    }
  }

  // Rebuild posMap with updated keys so subsequent lookups remain valid.
  BufferPosMap newPosMap;
  for (auto [oldVal, pos] : posMap) {
    auto it = allocRemap.find(oldVal);
    newPosMap[it != allocRemap.end() ? it->second : oldVal] = pos;
  }
  posMap = std::move(newPosMap);
}

} // namespace mlir::afir::buffer_placement

//===----------------------------------------------------------------------===//
// Annotation cleanup
//===----------------------------------------------------------------------===//

namespace {

/// Remove transient ascendc.* annotations from operations.
void clearAnnotations(func::FuncOp funcOp) {
  funcOp.walk([](Operation *op) {
    SmallVector<StringAttr> toRemove;
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName().getValue() == "ascendc.unit")
        continue;
      if (attr.getName().getValue() == "ascendc.kernel_kind")
        continue;
      if (attr.getName().getValue().starts_with("ascendc.")) {
        toRemove.push_back(attr.getName());
      }
    }
    for (StringAttr name : toRemove) {
      op->removeAttr(name);
    }
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace mlir::afir {

struct AscendCBufferPlacementPass
    : public ::impl::AscendCBufferPlacementPassBase<AscendCBufferPlacementPass> {
  using Base = ::impl::AscendCBufferPlacementPassBase<AscendCBufferPlacementPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto *ctx = &getContext();
    OpBuilder builder(ctx);
    IRRewriter rewriter(ctx);

    LLVM_DEBUG({
      llvm::dbgs() << "Running AscendCBufferPlacementPass on function: "
                   << funcOp.getName() << "\n";
    });

    // Step A: Parse all loop annotations
    auto loopAnns = collectLoopAnnotations(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "Found " << loopAnns.size() << " loop annotations\n";
    });

    // Step B: Infer buffer positions for existing allocs
    BufferPosMap posMap;
    inferBufferPositions(funcOp, loopAnns, posMap);

    LLVM_DEBUG({
      llvm::dbgs() << "Inferred positions for " << posMap.size()
                   << " allocs\n";
    });

    // Step C: Update function arguments to have GM memory_space
    annotateGMArgs(funcOp, builder);

    LLVM_DEBUG({
      llvm::dbgs() << "Updated function argument types\n";
    });

    // Step D: Propagate memory_space to subview results
    propagateSubviewMemorySpace(funcOp, rewriter);

    LLVM_DEBUG({
      llvm::dbgs() << "Propagated memory_space to subviews\n";
    });

    // Step E: Update alloc memory spaces (before insertCopiesForLoops so buffers have correct memory_space)
    updateAllocMemorySpace(posMap, rewriter);

    LLVM_DEBUG({
      llvm::dbgs() << "Updated alloc memory spaces\n";
    });

    // Re-propagate memory_space to subviews after alloc update
    // (e.g., CO1 alloc was updated, its subviews need to inherit the new memory_space)
    propagateSubviewMemorySpace(funcOp, rewriter);

    insertCopiesForLoops(funcOp, loopAnns, posMap, builder);

    LLVM_DEBUG({
      llvm::dbgs() << "Inserted copy operations\n";
    });

    // Step E2: Fix output buffer initialization.
    // The bufferizer produced memref.copy(src -> output_alloc) for allocs that
    // are used as accumulation buffers (CO1=7, VECOUT=10).
    // Replace with linalg.fill(0) so that K-axis iterations accumulate from
    // zero and so that the VECOUT tile is initialized correctly.
    // CO1 dealloc is also handled here (insert at end of allocating block).
    {
      struct OutputBufInfo {
        memref::AllocOp allocOp;
        SmallVector<memref::CopyOp> initCopies;
        SmallVector<memref::DeallocOp> existingDeallocs;
        bool isCo1; // true=CO1(7), false=VECOUT(10)
      };
      SmallVector<OutputBufInfo> outputBufInfos;
      funcOp.walk([&](memref::AllocOp allocOp) {
        auto memTy = cast<MemRefType>(allocOp.getType());
        auto memSpace = memTy.getMemorySpace();
        if (!memSpace)
          return;
        auto intAttr = dyn_cast<IntegerAttr>(memSpace);
        if (!intAttr)
          return;
        int64_t ms = intAttr.getInt();
        if (ms != 7 && ms != 10) // CO1=7, VECOUT=10
          return;

        Value buf = allocOp.getResult();
        Block *allocBlock = allocOp->getBlock();
        OutputBufInfo info;
        info.allocOp = allocOp;
        info.isCo1 = (ms == 7);
        for (Operation *user : llvm::make_early_inc_range(buf.getUsers())) {
          if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
            if (copyOp.getTarget() != buf)
              continue;
            if (copyOp->getBlock() != allocBlock)
              continue;
            // Skip fixpipe copies (src has memory_space > 0).
            Value src = copyOp.getSource();
            auto srcTy = dyn_cast<MemRefType>(src.getType());
            if (srcTy && srcTy.getMemorySpace())
              continue;
            info.initCopies.push_back(copyOp);
          } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
            info.existingDeallocs.push_back(deallocOp);
          }
        }
        outputBufInfos.push_back(std::move(info));
      });

      for (auto &info : outputBufInfos) {
        auto memTy = cast<MemRefType>(info.allocOp.getType());
        Value buf = info.allocOp.getResult();
        Block *allocBlock = info.allocOp->getBlock();

        // Replace init copies with linalg.fill(0).
        for (auto copyOp : info.initCopies) {
          builder.setInsertionPoint(copyOp);
          Value zero = builder.create<arith::ConstantOp>(
              copyOp.getLoc(), memTy.getElementType(),
              builder.getZeroAttr(memTy.getElementType()));
          builder.create<linalg::FillOp>(copyOp.getLoc(),
                                         ValueRange{zero}, ValueRange{buf});
          copyOp->erase();
        }

        // For CO1: remove bufferizer deallocs and insert a single correct one.
        // For VECOUT: the existing dealloc position is already correct (it was
        // placed by the bufferizer at the right spot), so leave it in place.
        if (info.isCo1) {
          for (auto deallocOp : info.existingDeallocs)
            deallocOp->erase();
          builder.setInsertionPoint(allocBlock->getTerminator());
          builder.create<memref::DeallocOp>(info.allocOp.getLoc(), buf);
        }
      }
    }

    // Step E3: Replace zero-buffer ins[1] of relu (max_signed) with a local
    // VECIN alloc + fill(0).
    //
    // The bufferizer generates a function-scoped %alloc (no memory_space) +
    // linalg.fill(0) for the relu "zeros" argument.  The pass must replace
    // each use inside the innermost loop with a fresh tile-sized VECIN alloc
    // (9:i32) + fill(0), and remove the function-scoped alloc/fill when all
    // uses have been replaced.
    {
      // Collect (elementwiseOp, operandIdx) pairs where the operand traces to
      // a function-scoped zero alloc (no memory_space, filled with 0).
      struct ZeroUse {
        linalg::ElementwiseOp elemOp;
        unsigned operandIdx;
        memref::AllocOp zeroAlloc; // the function-scoped zero alloc
      };
      SmallVector<ZeroUse> zeroUses;
      // Also track function-scoped zero allocs to clean up later.
      DenseSet<Operation *> zeroAllocsToClean;

      funcOp.walk([&](linalg::ElementwiseOp elemOp) {
        if (elemOp.getKind() != linalg::ElementwiseKind::max_signed)
          return;
        auto inputs = elemOp.getInputs();
        for (unsigned i = 0; i < inputs.size(); ++i) {
          Value inp = inputs[i];
          auto memTy = dyn_cast<MemRefType>(inp.getType());
          if (!memTy || memTy.getMemorySpace())
            continue; // already has memory_space, skip
          // Trace through subviews to the defining alloc.
          Value allocVal = traceToAlloc(inp);
          if (!allocVal)
            continue;
          auto allocOp = dyn_cast<memref::AllocOp>(allocVal.getDefiningOp());
          if (!allocOp)
            continue;
          // Check that the alloc is initialized with fill(0) (zero buffer).
          bool isFillZero = false;
          for (Operation *user : allocOp.getResult().getUsers()) {
            if (auto fillOp = dyn_cast<linalg::FillOp>(user)) {
              if (fillOp.getOutputs().front() == allocOp.getResult()) {
                // Check fill value is zero.
                if (auto constOp = dyn_cast<arith::ConstantOp>(
                        fillOp.getInputs().front().getDefiningOp())) {
                  if (constOp.getValue() ==
                      builder.getZeroAttr(
                          cast<MemRefType>(allocOp.getType()).getElementType()))
                    isFillZero = true;
                }
              }
            }
          }
          if (!isFillZero)
            continue;
          zeroUses.push_back({elemOp, i, allocOp});
          zeroAllocsToClean.insert(allocOp);
        }
      });

      for (auto &zu : zeroUses) {
        linalg::ElementwiseOp elemOp = zu.elemOp;
        // Determine tile shape from the outs operand (it has the correct shape).
        Value outsVal = elemOp.getOutputs().front();
        auto outsTy = cast<MemRefType>(outsVal.getType());

        OpBuilder localBuilder(elemOp);
        auto elemTy = cast<MemRefType>(zu.zeroAlloc.getType()).getElementType();
        // Create VECIN alloc with same dynamic sizes as outs.
        SmallVector<Value> dynSizes;
        for (unsigned d = 0; d < (unsigned)outsTy.getRank(); ++d) {
          if (outsTy.isDynamicDim(d)) {
            dynSizes.push_back(localBuilder.create<memref::DimOp>(
                elemOp.getLoc(), outsVal, d));
          }
        }
        auto vecinTy = MemRefType::get(outsTy.getShape(), elemTy,
                                       MemRefLayoutAttrInterface{},
                                       localBuilder.getI32IntegerAttr(9));
        Value vecinAlloc = localBuilder.create<memref::AllocOp>(
            elemOp.getLoc(), vecinTy, dynSizes);
        Value zero = localBuilder.create<arith::ConstantOp>(
            elemOp.getLoc(), elemTy, builder.getZeroAttr(elemTy));
        localBuilder.create<linalg::FillOp>(elemOp.getLoc(),
                                            ValueRange{zero},
                                            ValueRange{vecinAlloc});
        // Redirect the operand.
        elemOp.getInputsMutable()[zu.operandIdx].assign(vecinAlloc);
        // Dealloc after the elementwise op.
        OpBuilder afterBuilder(elemOp->getNextNode());
        afterBuilder.create<memref::DeallocOp>(elemOp.getLoc(), vecinAlloc);
      }

      // Remove function-scoped zero allocs and their fill + subview chains
      // if all uses have been replaced.
      // We must erase in reverse topological order (leaves first).
      for (Operation *allocOpBase : zeroAllocsToClean) {
        auto allocOp = cast<memref::AllocOp>(allocOpBase);
        Value allocVal = allocOp.getResult();
        // DFS: collect ops in post-order (children before parents) so we can
        // erase them leaf-first without use-after-free.
        SmallVector<Operation *> postOrder;
        bool safeToRemove = true;
        std::function<void(Value)> collectPostOrder = [&](Value v) {
          for (Operation *user : llvm::make_early_inc_range(v.getUsers())) {
            if (!safeToRemove)
              return;
            if (isa<linalg::FillOp, memref::DeallocOp, memref::DimOp>(user)) {
              postOrder.push_back(user);
            } else if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
              // Recurse into subview users first (post-order).
              collectPostOrder(sv.getResult());
              postOrder.push_back(user);
            } else {
              safeToRemove = false;
            }
          }
        };
        collectPostOrder(allocVal);
        if (!safeToRemove)
          continue;
        // Erase leaves first (post-order ensures children before parents).
        for (Operation *op : postOrder)
          op->erase();
        allocOp->erase();
      }
    }

    // Step F: Clear all ascendc.* annotations
    clearAnnotations(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "Cleared annotations\n";
    });

    // Step H: Verify IR
    if (failed(mlir::verify(funcOp))) {
      funcOp->emitError("Failed to verify after AscendCBufferPlacementPass");
      signalPassFailure();
    }
  }
};

} // namespace mlir::afir

//===----------------------------------------------------------------------===//
// Pass factory
//===----------------------------------------------------------------------===//

namespace mlir::afir {

std::unique_ptr<Pass> createAscendCBufferPlacementPass() {
  return std::make_unique<AscendCBufferPlacementPass>();
}

} // namespace mlir::afir
