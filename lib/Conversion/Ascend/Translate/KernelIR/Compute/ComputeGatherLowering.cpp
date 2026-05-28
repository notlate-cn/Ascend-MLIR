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

#include "ComputeLoweringInternal.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/ElementwiseBodyOpRegistry.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include <algorithm>
#include <limits>
#include <optional>

#define DEBUG_TYPE "ascend-compute-lower-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace ascend {

LogicalResult lowerGatherCompute(ComputeLoweringContext &lowering,
                                 linalg::GenericOp genOp,
                                 ArrayRef<linalg::GenericOp> parallelGenericOps) {
  if (!genOp->hasAttr(ascend::kGatherDimAttr))
    return failure();

  OpBuilder &builder = lowering.builder;
  Value outMemref = genOp.getDpsInitOperand(0)->get();
  // ---- Index-select gather: emit gather_l2 row by row ----
  // New pattern: {gather_dim = 1} attribute, 1 input (indices),
  // data accessed via memref.load in the body (captures a GM memref).
  // For each row i in 0..Tb_M:
  //   copy data row from GM → VECCALC
  //   gather_l2(dst_row[K], src_row[N], indices[K], 0, K)
  Value indicesMemref = genOp.getDpsInputOperand(0)->get();
  Location loc = genOp.getLoc();
  builder.setInsertionPoint(genOp);

  // Find data memref from memref.load in the body (bufferized tensor.extract).
  Value dataMemref;
  genOp.getBody()->walk([&](memref::LoadOp loadOp) {
    if (!dataMemref)
      dataMemref = loadOp.getMemref();
  });
  if (!dataMemref) {
    LLVM_DEBUG(llvm::dbgs()
               << "isIndexSelectGeneric: no memref.load found in body\n");
    return success();
  }

  // Find pre-gather op: parallel generic whose output memref == dataMemref
  // (i.e., the op that wrote the data we're gathering from)
  linalg::GenericOp preOp;
  for (linalg::GenericOp candidate : parallelGenericOps) {
    if (candidate == genOp) continue;
    if (candidate->hasAttr(ascend::kGatherDimAttr) ||
        candidate->hasAttr(ascend::kEmbeddingDimAttr))
      continue;
    if (candidate.getDpsInitOperand(0)->get() == dataMemref) {
      preOp = candidate;
      break;
    }
  }

  // Find post-gather op: parallel generic that has outMemref as one of its inputs
  linalg::GenericOp postOp;
  for (linalg::GenericOp candidate : parallelGenericOps) {
    if (candidate == genOp) continue;
    if (candidate->hasAttr(ascend::kGatherDimAttr) ||
        candidate->hasAttr(ascend::kEmbeddingDimAttr))
      continue;
    for (OpOperand *inp : candidate.getDpsInputOperands()) {
      if (inp->get() == outMemref) {
        postOp = candidate;
        break;
      }
    }
    if (postOp) break;
  }

  auto outMrt  = cast<MemRefType>(outMemref.getType());
  Type elemType = outMrt.getElementType();
  Type i32Type  = builder.getI32Type();

  // Tb_M = dim[0] of output, N = dim[1] of data, K = dim[1] of output
  Value tbM  = getDynDim(lowering, builder, loc, outMemref, 0);
  Value dimN = getDynDim(lowering, builder, loc, dataMemref, 1);
  Value dimK = getDynDim(lowering, builder, loc, outMemref, 1);

  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;

  // Get indices as a local_tensor.
  // If indices are in VECIN (ms=9), deque from queue.
  // If indices are in GM (ms=0), copy into VECCALC first.
  auto idxMrt = cast<MemRefType>(indicesMemref.getType());
  Type idxElemType = idxMrt.getElementType();
  Value idxCount = getDynDim(lowering, builder, loc, indicesMemref, 0);
  Value indicesLt;
  int64_t idxMs = getMemorySpace(indicesMemref.getType());
  if (idxMs == 9 /*VECIN*/ || idxMs == 11 /*VECCALC*/) {
    indicesLt = lowering.readTensor(builder, loc, indicesMemref);
  } else {
    // GM: copy indices into a fresh VECCALC buffer.
    SmallVector<Value> idxDims = {idxCount};
    auto [idxTbuf, idxLt] =
        allocVeccalc(lowering, builder, loc, idxElemType, idxDims);
    Value idxGt = builder.create<GlobalTensorOp>(
        loc, GlobalTensorType::get(idxElemType));
    builder.create<GlobalTensorSetGlobalBufferOp>(loc, idxGt, indicesMemref,
                                                   /*size=*/Value{});
    builder.create<DataCopyL2Op>(loc, idxLt, idxGt, idxCount);
    indicesLt = idxLt;
  }

  Value dimK_i32 =
      builder.create<arith::IndexCastOp>(loc, i32Type, dimK);
  Value srcBaseAddr =
      builder.create<arith::ConstantIntOp>(loc, i32Type, 0);

  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value outTbuf = lowering.ctx.getTBuf(outMemref);
  if (!outTbuf) {
    genOp.emitError("missing TBuf for gather output buffer");
    return failure();
  }

  // Byte size for a single output row (K * elemBytes).
  Value elemBytesVal =
      builder.create<arith::ConstantIndexOp>(loc, elemBytes);
  Value outBytesPerRow =
      builder.create<arith::MulIOp>(loc, dimK, elemBytesVal);

  // Collect enclosing scf.for induction variables to compute the global
  // row offset into the data memref.  The generic sits inside nested
  // tiling loops whose IVs sum to the tile origin in the data tensor.
  SmallVector<Value> enclosingIVs;
  for (Operation *p = genOp->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p))
      enclosingIVs.push_back(f.getInductionVar());

  // Data is in GM: set up a GlobalTensor for row-by-row copy.
  Value dataGt = builder.create<GlobalTensorOp>(
      loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, dataGt, dataMemref,
                                                 /*size=*/Value{});

  // Allocate a VECCALC buffer for one data row plus one Gather vector
  // window of padding. AscendC Gather offsets are byte offsets into UB; on
  // hardware the vector instruction can still bounds-check a full element
  // window past the logical index. Padding keeps max-index gathers inside
  // the registered UB buffer.
  unsigned gatherPadElems =
      elemBytes <= 2 ? 256 : (elemBytes <= 4 ? 128 : 64);
  unsigned gatherChunkElems =
      elemBytes <= 2 ? 128 : (elemBytes <= 4 ? 64 : 32);
  Value gatherPadElemsVal =
      builder.create<arith::ConstantIndexOp>(loc, gatherPadElems);
  Value paddedDimN =
      builder.create<arith::AddIOp>(loc, dimN, gatherPadElemsVal);
  Value paddedDimK =
      ceilToMultipleIndex(builder, loc, dimK, gatherChunkElems);
  Value dataRowQueue = builder.create<QueueOp>(
      loc, QueueType::get(lowering.mlirCtx, TPosition::VECIN, 1));
  Value rowBytes = builder.create<arith::MulIOp>(
      loc, paddedDimN,
      builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value dataRowQueueDepth =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
  builder.create<TPipeInitQueueOp>(loc, lowering.ctx.pipe, dataRowQueue,
                                   dataRowQueueDepth, rowBytes);

  // Gather and post-gather vector ops must run in VECCALC.  Real hardware
  // rejects some VEC reads/writes against VECOUT TBuf slices that the
  // simulator accepts, so rows are copied to VECOUT only after vector work.
  auto gatheredRowAlloc =
      allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{paddedDimK});
  Value gatheredRowLt = gatheredRowAlloc.second;
  auto gatherSourceRowAlloc =
      allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{paddedDimN});
  Value gatherSourceRowLt = gatherSourceRowAlloc.second;

  // Pre-op temporaries are reused for every row. Initializing these TPipe
  // buffers inside the row loop exhausts simulator buffer bookkeeping for
  // larger M even though the loop is sequential.
  Value preProcessedRowLt;
  Value preDimN_i32;
  llvm::SmallDenseMap<Value, Value> preInvariantConstLt;
  if (preOp) {
    auto [procTbuf, procLt] =
        allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{paddedDimN});
    (void)procTbuf;
    preProcessedRowLt = procLt;
    preDimN_i32 =
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), dimN);

    Block &preBody = *preOp.getBody();
    for (auto &bodyOp : preBody.without_terminator()) {
      for (Value operand : bodyOp.getOperands()) {
        auto constOp = operand.getDefiningOp<arith::ConstantOp>();
        if (!constOp || preInvariantConstLt.contains(operand))
          continue;
        auto [dupTbuf, dupLt] =
            allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{dimN});
        (void)dupTbuf;
        auto dupOp = builder.create<DuplicateL2Op>(
            loc, dupLt, constOp.getResult(), preDimN_i32);
        lowering.copyAscendCUnitAttr(preOp.getOperation(), dupOp.getOperation());
        preInvariantConstLt[operand] = dupLt;
      }
    }
  }

  Value fusedBodyDimK_i32;
  llvm::SmallDenseMap<Value, Value> fusedBodyInvariantConstLt;
  llvm::SmallDenseMap<Value, Value> fusedBodyInvariantInputLt;
  if (!postOp) {
    fusedBodyDimK_i32 =
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), dimK);
    Block &gatherBody = *genOp.getBody();
    unsigned numBodyIns = static_cast<unsigned>(genOp.getNumDpsInputs());

    for (unsigned argNum = 1; argNum < numBodyIns; ++argNum) {
      BlockArgument blockArg = gatherBody.getArgument(argNum);
      Value argMemref = genOp.getDpsInputOperand(argNum)->get();
      if (argMemref == outMemref)
        continue;
      int64_t argMs = getMemorySpace(argMemref.getType());
      if (argMs == 0) {
        auto argMrt = cast<MemRefType>(argMemref.getType());
        Type argElem = argMrt.getElementType();
        Value argGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(argElem));
        builder.create<GlobalTensorSetGlobalBufferOp>(
            loc, argGt, argMemref, /*size=*/Value{});
        Value argCount = computeElementCount(builder, loc, argMemref);
        fusedBodyInvariantInputLt[blockArg] =
            copyGmToVeccalc(lowering, builder, loc, argElem, argGt, argCount);
      }
    }

    bool pastLoad = false;
    for (auto &bodyOp : gatherBody.without_terminator()) {
      if (isa<memref::LoadOp>(bodyOp)) {
        pastLoad = true;
        continue;
      }
      if (!pastLoad)
        continue;
      for (Value operand : bodyOp.getOperands()) {
        auto constOp = operand.getDefiningOp<arith::ConstantOp>();
        if (!constOp || fusedBodyInvariantConstLt.contains(operand))
          continue;
        auto [dupTbuf, dupLt] =
            allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{dimK});
        (void)dupTbuf;
        auto dupOp = builder.create<DuplicateL2Op>(
            loc, dupLt, constOp.getResult(), fusedBodyDimK_i32);
        lowering.copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
        fusedBodyInvariantConstLt[operand] = dupLt;
      }
    }
  }

  builder.create<scf::ForOp>(
      loc, zero, tbM, one, ValueRange{},
      [&](OpBuilder &b, Location forLoc, Value rowIdx, ValueRange) {
        // Global row = sum(enclosing IVs) + rowIdx (tile-local row)
        Value globalRow = rowIdx;
        for (Value iv : enclosingIVs)
          globalRow = b.create<arith::AddIOp>(forLoc, globalRow, iv);

        // Step 1: Copy one data row from GM → VECCALC.
        // Offset the GlobalTensor by globalRow * N elements, then DataCopy.
        Value rowElemOff =
            b.create<arith::MulIOp>(forLoc, globalRow, dimN);
        Value dataRowGt = b.create<GlobalTensorBracketOp>(
            forLoc, GlobalTensorType::get(elemType), dataGt,
            rowElemOff);
        Value dataRowAllocLt = b.create<TQueBindAllocTensorOp>(
            forLoc, LocalTensorType::get(elemType), dataRowQueue);
        b.create<DataCopyL2Op>(forLoc, dataRowAllocLt, dataRowGt, dimN);
        b.create<TQueBindEnqueTensorOp>(forLoc, dataRowQueue,
                                        dataRowAllocLt);
        Value dataRowLt = b.create<TQueBindDequeTensorOp>(
            forLoc, LocalTensorType::get(elemType), dataRowQueue);

        // Step 1b: If pre-op exists (e.g. relu), apply it on dataRowLt
        Value processedRowLt = dataRowLt;
        if (preOp) {
          Value procLt = preProcessedRowLt;
          Value dimN_i32 = preDimN_i32;

          Block &preBody = *preOp.getBody();
          llvm::SmallDenseMap<Value, Value> preValToLt;

          auto preResolve = [&](Value v) -> Value {
            if (auto ba = dyn_cast<BlockArgument>(v)) {
              if (ba.getArgNumber() == 0) return dataRowLt;
              return procLt;
            }
            auto it = preValToLt.find(v);
            if (it != preValToLt.end()) return it->second;
            if (v.getDefiningOp<arith::ConstantOp>()) {
              auto constIt = preInvariantConstLt.find(v);
              if (constIt == preInvariantConstLt.end())
                return Value{};
              Value dupLt = constIt->second;
              preValToLt[v] = dupLt;
              return dupLt;
            }
            return Value{};
          };

          for (auto &bodyOp : preBody.without_terminator()) {
            if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
              Value lhs = preResolve(maxOp.getLhs()), rhs = preResolve(maxOp.getRhs());
              if (lhs && rhs) {
                auto maxOp2 = b.create<MaxL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                lowering.copyAscendCUnitAttr(preOp.getOperation(), maxOp2.getOperation());
                preValToLt[maxOp.getResult()] = procLt;
              }
            } else if (auto addOp2 = dyn_cast<arith::AddFOp>(bodyOp)) {
              Value lhs = preResolve(addOp2.getLhs()), rhs = preResolve(addOp2.getRhs());
              if (lhs && rhs) {
                auto addOp3 = b.create<AddL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                lowering.copyAscendCUnitAttr(preOp.getOperation(), addOp3.getOperation());
                preValToLt[addOp2.getResult()] = procLt;
              }
            } else if (auto mulOp2 = dyn_cast<arith::MulFOp>(bodyOp)) {
              Value lhs = preResolve(mulOp2.getLhs()), rhs = preResolve(mulOp2.getRhs());
              if (lhs && rhs) {
                auto mulOp3 = b.create<MulL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                lowering.copyAscendCUnitAttr(preOp.getOperation(), mulOp3.getOperation());
                preValToLt[mulOp2.getResult()] = procLt;
              }
            }
          }
          processedRowLt = procLt;
        } else {
          emitLocalToLocalScalarCopy(b, forLoc, elemType,
                                     gatherSourceRowLt, dataRowLt, dimN);
          processedRowLt = gatherSourceRowLt;
        }
        emitLocalTensorZeroPad(b, forLoc, elemType, processedRowLt, dimN,
                               paddedDimN);

        // Step 2: gather_l2(dst[K], src[N], indices, srcBase=0, count=K)
        Value dstByteOff =
            b.create<arith::MulIOp>(forLoc, rowIdx, outBytesPerRow);
        Value dstRowLt = b.create<TBufGetWithOffsetOp>(
            forLoc, LocalTensorType::get(elemType), outTbuf,
            dimK, dstByteOff);
        b.create<GatherL2Op>(forLoc, gatheredRowLt, processedRowLt,
                             indicesLt, srcBaseAddr, dimK_i32);

        // Step 3: If post-op exists (e.g. add bias), apply it on gatheredRowLt.
        // If no post-op, walk the gather body itself for any arith ops that
        // appear after the memref.load (from upstream fusion). This handles
        // the case where relu + add were fused into the gather body by
        // ascend-kernelize gather elementwise fusion before bufferization.
        if (postOp) {
          Value dimK_i32v = b.create<arith::IndexCastOp>(forLoc, b.getI32Type(), dimK);
          Block &postBody = *postOp.getBody();
          llvm::SmallDenseMap<Value, Value> postValToLt;

          auto postResolve = [&](Value v) -> Value {
            if (auto ba = dyn_cast<BlockArgument>(v)) {
              unsigned argNum = ba.getArgNumber();
              unsigned numIns = (unsigned)postOp.getNumDpsInputs();
              if (argNum >= numIns) return gatheredRowLt; // output init arg
              Value argMemref = postOp.getDpsInputOperand(argNum)->get();
              if (argMemref == outMemref) return gatheredRowLt;
              int64_t argMs = getMemorySpace(argMemref.getType());
              if (argMs == 0) {
                auto argMrt = cast<MemRefType>(argMemref.getType());
                Type argElem = argMrt.getElementType();
                Value argGt = b.create<GlobalTensorOp>(
                    forLoc, GlobalTensorType::get(argElem));
                b.create<GlobalTensorSetGlobalBufferOp>(
                    forLoc, argGt, argMemref, /*size=*/Value{});
                Value argCount = computeElementCount(b, forLoc, argMemref);
                return copyGmToVeccalc(lowering, b, forLoc, argElem, argGt, argCount);
              }
              // Other on-chip inputs (bias, etc.) — read their tensor.
              return lowering.readTensor(b, forLoc, argMemref);
            }
            auto it = postValToLt.find(v);
            if (it != postValToLt.end()) return it->second;
            if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
              auto [dupTbuf3, dupLt] = allocVeccalc(lowering, b, forLoc, elemType,
                                                     SmallVector<Value>{dimK});
              auto dupOp3 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimK_i32v);
              lowering.copyAscendCUnitAttr(postOp.getOperation(), dupOp3.getOperation());
              postValToLt[v] = dupLt;
              return dupLt;
            }
            return Value{};
          };

          for (auto &bodyOp : postBody.without_terminator()) {
            if (auto addOp3 = dyn_cast<arith::AddFOp>(bodyOp)) {
              Value lhs = postResolve(addOp3.getLhs()), rhs = postResolve(addOp3.getRhs());
              if (lhs && rhs) {
                auto addOp4 = b.create<AddL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                lowering.copyAscendCUnitAttr(postOp.getOperation(), addOp4.getOperation());
                postValToLt[addOp3.getResult()] = gatheredRowLt;
              }
            } else if (auto mulOp3 = dyn_cast<arith::MulFOp>(bodyOp)) {
              Value lhs = postResolve(mulOp3.getLhs()), rhs = postResolve(mulOp3.getRhs());
              if (lhs && rhs) {
                auto mulOp4 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                lowering.copyAscendCUnitAttr(postOp.getOperation(), mulOp4.getOperation());
                postValToLt[mulOp3.getResult()] = gatheredRowLt;
              }
            } else if (auto maxOp3 = dyn_cast<arith::MaximumFOp>(bodyOp)) {
              Value lhs = postResolve(maxOp3.getLhs()), rhs = postResolve(maxOp3.getRhs());
              if (lhs && rhs) {
                auto maxOp4 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                lowering.copyAscendCUnitAttr(postOp.getOperation(), maxOp4.getOperation());
                postValToLt[maxOp3.getResult()] = gatheredRowLt;
              }
            }
          }
        } else {
          // Walk the fused gather body for arith ops that appear after
          // the memref.load (these were inlined by gather elementwise fusion).
          // Block args:
          //   arg0 = indices element (i64, skip)
          //   arg1..argN-2 = extra ins (bias etc.)
          //   argN-1 = out init (skip, use gatheredRowLt instead)
          Value dimK_i32v = fusedBodyDimK_i32;
          Block &gatherBody = *genOp.getBody();
          unsigned numBodyIns = (unsigned)genOp.getNumDpsInputs();
          llvm::SmallDenseMap<Value, Value> bodyValToLt;

          // Helper: find the memref.load result in the body.
          Value loadResult;
          for (auto &op : gatherBody.without_terminator()) {
            if (isa<memref::LoadOp>(op)) {
              loadResult = op.getResult(0);
              break;
            }
          }

          auto bodyResolve = [&](Value v) -> Value {
            // The "gathered row" value — the memref.load result maps to
            // gatheredRowLt (post-gather result).
            if (loadResult && v == loadResult) return gatheredRowLt;
            auto it = bodyValToLt.find(v);
            if (it != bodyValToLt.end()) return it->second;
            if (auto ba = dyn_cast<BlockArgument>(v)) {
              unsigned argNum = ba.getArgNumber();
              if (argNum == 0) return Value{}; // indices arg, skip
              if (argNum >= numBodyIns) return gatheredRowLt; // out init
              if (auto inputIt = fusedBodyInvariantInputLt.find(ba);
                  inputIt != fusedBodyInvariantInputLt.end()) {
                bodyValToLt[v] = inputIt->second;
                return inputIt->second;
              }
              // Extra ins (bias, etc.) at argNum=1..numBodyIns-1
              Value argMemref = genOp.getDpsInputOperand(argNum)->get();
              int64_t argMs = getMemorySpace(argMemref.getType());
              if (argMs > 0) {
                // On-chip: use readTensor directly.
                Value lt = lowering.readTensor(b, forLoc, argMemref);
                bodyValToLt[v] = lt;
                return lt;
              }
              // GM: copy to VECCALC for vector ops. The CANN translation
              // rewrites the narrow gather+bias Add to read the copied
              // scalar source directly from GM in the simulator.
              Value argCount = computeElementCount(b, forLoc, argMemref);
              auto argMrt = cast<MemRefType>(argMemref.getType());
              Type argElem = argMrt.getElementType();
              Value argGt = b.create<GlobalTensorOp>(forLoc, GlobalTensorType::get(argElem));
              b.create<GlobalTensorSetGlobalBufferOp>(forLoc, argGt, argMemref,
                                                       /*size=*/Value{});
              Value argLt =
                  copyGmToVeccalc(lowering, b, forLoc, argElem, argGt, argCount);
              bodyValToLt[v] = argLt;
              return argLt;
            }
            if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
              auto constIt = fusedBodyInvariantConstLt.find(v);
              if (constIt == fusedBodyInvariantConstLt.end())
                return Value{};
              Value dupLt = constIt->second;
              bodyValToLt[v] = dupLt;
              return dupLt;
            }
            return Value{};
          };

          bool pastLoad = false;
          for (auto &op : gatherBody.without_terminator()) {
            if (isa<memref::LoadOp>(op)) {
              pastLoad = true;
              continue;
            }
            if (!pastLoad) continue;
            if (auto addOp4 = dyn_cast<arith::AddFOp>(op)) {
              Value lhs = bodyResolve(addOp4.getLhs()),
                    rhs = bodyResolve(addOp4.getRhs());
              if (lhs && rhs) {
                auto addOp5 = b.create<AddL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                lowering.copyAscendCUnitAttr(genOp.getOperation(), addOp5.getOperation());
                bodyValToLt[addOp4.getResult()] = gatheredRowLt;
              }
            } else if (auto maxOp4 = dyn_cast<arith::MaximumFOp>(op)) {
              Value lhs = bodyResolve(maxOp4.getLhs()),
                    rhs = bodyResolve(maxOp4.getRhs());
              if (lhs && rhs) {
                auto maxOp5 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                lowering.copyAscendCUnitAttr(genOp.getOperation(), maxOp5.getOperation());
                bodyValToLt[maxOp4.getResult()] = gatheredRowLt;
              }
            } else if (auto mulOp4 = dyn_cast<arith::MulFOp>(op)) {
              Value lhs = bodyResolve(mulOp4.getLhs()),
                    rhs = bodyResolve(mulOp4.getRhs());
              if (lhs && rhs) {
                auto mulOp5 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                lowering.copyAscendCUnitAttr(genOp.getOperation(), mulOp5.getOperation());
                bodyValToLt[mulOp4.getResult()] = gatheredRowLt;
              }
            }
          }
        }

        emitLocalToLocalScalarCopy(b, forLoc, elemType, dstRowLt,
                                   gatheredRowLt, dimK);
        b.create<TQueBindFreeTensorOp>(forLoc, dataRowQueue, dataRowLt);
        b.create<scf::YieldOp>(forLoc);
      });

  if (postOp) postOp.erase();
  if (preOp) preOp.erase();
  genOp.erase();
  return success();
}

} // namespace ascend
} // namespace mlir
