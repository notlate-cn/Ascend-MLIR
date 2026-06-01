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

#include "Conversion/LinalgToAscendC/ComputeConversionContext.h"
#include "Conversion/LinalgToAscendC/ComputeConversionHelpers.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCUtils.h"

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
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include <algorithm>

#define DEBUG_TYPE "linalg-to-ascendc-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

LogicalResult convertParallelGenerics(func::FuncOp funcOp, ComputeCtx &cc) {
  [[maybe_unused]] OpBuilder &builder = cc.builder;
  [[maybe_unused]] AscendCBufferContext &ctx = cc.ctx;
  [[maybe_unused]] MLIRContext *mlirCtx = cc.mlirCtx;

  // --- linalg.generic {all-parallel, on-chip output} ---
  //
  // Pure-parallel generic lowering (e.g. broadcast+add, broadcast+mul).
  // These have iterator_types = ["parallel", "parallel", ...] with no reduction.
  //
  // Strategy mirrors the reduction path (Steps 1-2) but skips Step 3:
  //   GM/VECIN inputs → promote to VECCALC local_tensors (broadcast_l2 or copy)
  //   Body arith ops → inline as AscendC vector ops on VECCALC accumulator
  //   Final result   → write directly to VECOUT (writeTensor handles alloc)
  //   Enqueue VECOUT for downstream data-move epilogue copy
  //
  // Concat semantics are implicitly handled: the VECOUT→GM copy op (inserted by
  // AscendCBufferPlacement + DataMoveConversion) targets a memref subview of the
  // output buffer with the correct byte offset, so Op1 and Op2 results land at
  // the right positions in the concatenated output without any asc.concat op.
  SmallVector<linalg::GenericOp> parallelGenericOps;
  funcOp.walk([&](linalg::GenericOp op) {
    auto iterTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iterTypes, [](utils::IteratorType t) {
      return t == utils::IteratorType::parallel;
    });
    if (allParallel && op.getNumDpsInits() == 1)
      parallelGenericOps.push_back(op);
  });

  for (linalg::GenericOp genOp : parallelGenericOps) {
    Value outMemref = genOp.getDpsInitOperand(0)->get();
    int64_t outMs   = getMemorySpace(outMemref.getType());
    if (outMs <= 0)
      continue; // output must be on-chip (VECOUT or VECCALC)

    // ---- Index-select gather: emit gather_l2 row by row ----
    // New pattern: {gather_dim = 1} attribute, 1 input (indices),
    // data accessed via memref.load in the body (captures a GM memref).
    // For each row i in 0..Tb_M:
    //   copy data row from GM → VECCALC
    //   gather_l2(dst_row[K], src_row[N], indices[K], 0, K)
    if (isIndexSelectGeneric(genOp)) {
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
        continue;
      }

      // Find pre-gather op: parallel generic whose output memref == dataMemref
      // (i.e., the op that wrote the data we're gathering from)
      linalg::GenericOp preOp;
      for (linalg::GenericOp candidate : parallelGenericOps) {
        if (candidate == genOp) continue;
        if (candidate->hasAttr("gather_dim") || candidate->hasAttr("embedding_dim")) continue;
        if (candidate.getDpsInitOperand(0)->get() == dataMemref) {
          preOp = candidate;
          break;
        }
      }

      // Find post-gather op: parallel generic that has outMemref as one of its inputs
      linalg::GenericOp postOp;
      for (linalg::GenericOp candidate : parallelGenericOps) {
        if (candidate == genOp) continue;
        if (candidate->hasAttr("gather_dim") || candidate->hasAttr("embedding_dim")) continue;
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
      Value tbM  = cc.getDynDim(builder, loc, outMemref, 0);
      Value dimN = cc.getDynDim(builder, loc, dataMemref, 1);
      Value dimK = cc.getDynDim(builder, loc, outMemref, 1);

      unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;

      // Get indices as a local_tensor.
      // If indices are in VECIN (ms=9), deque from queue.
      // If indices are in GM (ms=0), copy into VECCALC first.
      auto idxMrt = cast<MemRefType>(indicesMemref.getType());
      Type idxElemType = idxMrt.getElementType();
      Value idxCount = cc.getDynDim(builder, loc, indicesMemref, 0);
      Value indicesLt;
      int64_t idxMs = getMemorySpace(indicesMemref.getType());
      if (idxMs == 9 /*VECIN*/ || idxMs == 11 /*VECCALC*/) {
        indicesLt = cc.readTensor(builder, loc, indicesMemref);
      } else {
        // GM: copy indices into a fresh VECCALC buffer.
        SmallVector<Value> idxDims = {idxCount};
        auto [idxTbuf, idxLt] =
            cc.allocVeccalc(builder, loc, idxElemType, idxDims);
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

      // Alloc the VECOUT output tensor.
      Value dstLt = cc.writeTensor(builder, loc, outMemref);

      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);
      // Element offset for a single output row.
      Value outElemsPerRow = dimK;

      // Collect enclosing scf.for induction variables to compute the global
      // row offset into the data memref.  The generic sits inside nested
      // tiling loops whose IVs sum to the tile origin in the data tensor.
      SmallVector<Value> enclosingIVs;
      for (Operation *p = genOp->getParentOp(); p; p = p->getParentOp())
        if (auto f = dyn_cast<scf::ForOp>(p))
          enclosingIVs.push_back(f.getInductionVar());

      // Set by the gather-body dispatch loops below when an op is encountered
      // that isn't in their supported set; checked after the scf.for build to
      // fail loudly (mirrors the elementwise loops at lines ~830, ~1808).
      // Without this, an unhandled op (e.g. select+cmpf before 998edb7) is
      // silently dropped and the kernel computes the wrong thing.
      Operation *gatherUnsupportedOp = nullptr;

      // Data is in GM: set up a GlobalTensor for row-by-row copy.
      Value dataGt = builder.create<GlobalTensorOp>(
          loc, GlobalTensorType::get(elemType));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dataGt, dataMemref,
                                                     /*size=*/Value{});

      // Allocate a VECCALC buffer for one data row plus one 32B datablock of
      // padding.  AscendC Gather offsets are byte offsets into UB; on hardware
      // the vector instruction may touch the tail datablock even when the
      // logical element is within [0, N).  Padding keeps max-index gathers from
      // reading past the registered UB buffer.
      unsigned gatherPadElems = std::max<unsigned>(1, 32 / elemBytes);
      Value gatherPadElemsVal =
          builder.create<arith::ConstantIndexOp>(loc, gatherPadElems);
      Value paddedDimN =
          builder.create<arith::AddIOp>(loc, dimN, gatherPadElemsVal);
      Value dataRowQueue = builder.create<QueueOp>(
          loc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
      Value rowBytes = builder.create<arith::MulIOp>(
          loc, paddedDimN,
          builder.create<arith::ConstantIndexOp>(loc, elemBytes));
      Value dataRowQueueDepth =
          builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
      builder.create<TPipeInitQueueOp>(loc, ctx.pipe, dataRowQueue,
                                       dataRowQueueDepth, rowBytes);

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
              Value procLt =
                  cc.allocVeccalc(b, forLoc, elemType, SmallVector<Value>{dimN})
                      .second;
              Value dimN_i32 = b.create<arith::IndexCastOp>(forLoc, b.getI32Type(), dimN);

              Block &preBody = *preOp.getBody();
              llvm::SmallDenseMap<Value, Value> preValToLt;

              auto preResolve = [&](Value v) -> Value {
                if (auto ba = dyn_cast<BlockArgument>(v)) {
                  if (ba.getArgNumber() == 0) return dataRowLt;
                  return procLt;
                }
                auto it = preValToLt.find(v);
                if (it != preValToLt.end()) return it->second;
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf2, dupLt] = cc.allocVeccalc(b, forLoc, elemType,
                                                         SmallVector<Value>{dimN});
                  auto dupOp2 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimN_i32);
                  copyAscendCUnitAttr(preOp.getOperation(), dupOp2.getOperation());
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
                    copyAscendCUnitAttr(preOp.getOperation(), maxOp2.getOperation());
                    preValToLt[maxOp.getResult()] = procLt;
                  }
                } else if (auto addOp2 = dyn_cast<arith::AddFOp>(bodyOp)) {
                  Value lhs = preResolve(addOp2.getLhs()), rhs = preResolve(addOp2.getRhs());
                  if (lhs && rhs) {
                    auto addOp3 = b.create<AddL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    copyAscendCUnitAttr(preOp.getOperation(), addOp3.getOperation());
                    preValToLt[addOp2.getResult()] = procLt;
                  }
                } else if (auto mulOp2 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = preResolve(mulOp2.getLhs()), rhs = preResolve(mulOp2.getRhs());
                  if (lhs && rhs) {
                    auto mulOp3 = b.create<MulL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    copyAscendCUnitAttr(preOp.getOperation(), mulOp3.getOperation());
                    preValToLt[mulOp2.getResult()] = procLt;
                  }
                } else if (!isa<arith::ConstantOp>(bodyOp)) {
                  if (!gatherUnsupportedOp) gatherUnsupportedOp = &bodyOp;
                }
              }
              processedRowLt = procLt;
            }

            // Step 2: gather_l2(dst[K], src[N], indices, srcBase=0, count=K)
            Value dstElemOff =
                b.create<arith::MulIOp>(forLoc, rowIdx, outElemsPerRow);
            Value dstRowLt = b.create<LocalTensorSubIndexOp>(
                forLoc, LocalTensorType::get(elemType), dstLt, dstElemOff);
            Value gatheredRowLt =
                cc.allocVeccalc(b, forLoc, elemType, SmallVector<Value>{dimK})
                    .second;
            b.create<GatherL2Op>(forLoc, gatheredRowLt, processedRowLt,
                                 indicesLt, srcBaseAddr, dimK_i32);

            // Step 3: If post-op exists (e.g. add bias), apply it on gatheredRowLt.
            // If no post-op, walk the gather body itself for any arith ops that
            // appear after the memref.load (from upstream fusion). This handles
            // the case where relu + add were fused into the gather body by
            // --fuse-gather-elementwise before bufferization.
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
                  // Other inputs (bias, etc.) — read their tensor
                  return cc.readTensor(b, forLoc, argMemref);
                }
                auto it = postValToLt.find(v);
                if (it != postValToLt.end()) return it->second;
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf3, dupLt] = cc.allocVeccalc(b, forLoc, elemType,
                                                         SmallVector<Value>{dimK});
                  auto dupOp3 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimK_i32v);
                  copyAscendCUnitAttr(postOp.getOperation(), dupOp3.getOperation());
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
                    copyAscendCUnitAttr(postOp.getOperation(), addOp4.getOperation());
                    postValToLt[addOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp3 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = postResolve(mulOp3.getLhs()), rhs = postResolve(mulOp3.getRhs());
                  if (lhs && rhs) {
                    auto mulOp4 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(postOp.getOperation(), mulOp4.getOperation());
                    postValToLt[mulOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp3 = dyn_cast<arith::MaximumFOp>(bodyOp)) {
                  Value lhs = postResolve(maxOp3.getLhs()), rhs = postResolve(maxOp3.getRhs());
                  if (lhs && rhs) {
                    auto maxOp4 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(postOp.getOperation(), maxOp4.getOperation());
                    postValToLt[maxOp3.getResult()] = gatheredRowLt;
                  }
                } else if (!isa<arith::ConstantOp>(bodyOp)) {
                  if (!gatherUnsupportedOp) gatherUnsupportedOp = &bodyOp;
                }
              }
            } else {
              // Walk the fused gather body for arith ops that appear after
              // the memref.load (these were inlined by --fuse-gather-elementwise).
              // Block args:
              //   arg0 = indices element (i64, skip)
              //   arg1..argN-2 = extra ins (bias etc.)
              //   argN-1 = out init (skip, use gatheredRowLt instead)
              Value dimK_i32v = b.create<arith::IndexCastOp>(forLoc, b.getI32Type(), dimK);
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

              SmallVector<std::pair<Value, Value>> bodyTempQueueTensors;

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
                  // Extra ins (bias, etc.) at argNum=1..numBodyIns-1
                  Value argMemref = genOp.getDpsInputOperand(argNum)->get();
                  int64_t argMs = getMemorySpace(argMemref.getType());
                  if (argMs > 0) {
                    // On-chip: use readTensor directly.
                    Value lt = cc.readTensor(b, forLoc, argMemref);
                    bodyValToLt[v] = lt;
                    return lt;
                  }
                  // GM: copy through a VECIN queue so the vector op observes
                  // a synchronized local tensor on real hardware.
                  Value argCount = cc.getDynDim(b, forLoc, argMemref, 0);
                  auto argMrt = cast<MemRefType>(argMemref.getType());
                  Type argElem = argMrt.getElementType();
                  Value argQueue = b.create<QueueOp>(
                      forLoc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
                  unsigned argElemBytes =
                      argElem.getIntOrFloatBitWidth() / 8;
                  Value argBytes = b.create<arith::MulIOp>(
                      forLoc, argCount,
                      b.create<arith::ConstantIndexOp>(forLoc, argElemBytes));
                  Value depth = b.create<arith::ConstantOp>(
                      forLoc, b.getI32IntegerAttr(1));
                  b.create<TPipeInitQueueOp>(forLoc, ctx.pipe, argQueue, depth,
                                             argBytes);
                  Value argAllocLt = b.create<TQueBindAllocTensorOp>(
                      forLoc, LocalTensorType::get(argElem), argQueue);
                  Value argGt = b.create<GlobalTensorOp>(forLoc, GlobalTensorType::get(argElem));
                  b.create<GlobalTensorSetGlobalBufferOp>(forLoc, argGt, argMemref,
                                                           /*size=*/Value{});
                  b.create<DataCopyL2Op>(forLoc, argAllocLt, argGt, argCount);
                  b.create<TQueBindEnqueTensorOp>(forLoc, argQueue, argAllocLt);
                  Value argLt = b.create<TQueBindDequeTensorOp>(
                      forLoc, LocalTensorType::get(argElem), argQueue);
                  bodyTempQueueTensors.push_back({argQueue, argLt});
                  bodyValToLt[v] = argLt;
                  return argLt;
                }
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf4, dupLt] = cc.allocVeccalc(b, forLoc, elemType,
                                                         SmallVector<Value>{dimK});
                  auto dupOp4 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimK_i32v);
                  copyAscendCUnitAttr(genOp.getOperation(), dupOp4.getOperation());
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
                    copyAscendCUnitAttr(genOp.getOperation(), addOp5.getOperation());
                    bodyValToLt[addOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp4 = dyn_cast<arith::MaximumFOp>(op)) {
                  Value lhs = bodyResolve(maxOp4.getLhs()),
                        rhs = bodyResolve(maxOp4.getRhs());
                  if (lhs && rhs) {
                    auto maxOp5 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(genOp.getOperation(), maxOp5.getOperation());
                    bodyValToLt[maxOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp4 = dyn_cast<arith::MulFOp>(op)) {
                  Value lhs = bodyResolve(mulOp4.getLhs()),
                        rhs = bodyResolve(mulOp4.getRhs());
                  if (lhs && rhs) {
                    auto mulOp5 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(genOp.getOperation(), mulOp5.getOperation());
                    bodyValToLt[mulOp4.getResult()] = gatheredRowLt;
                  }
                } else if (!isa<arith::ConstantOp, memref::LoadOp>(op)) {
                  if (!gatherUnsupportedOp) gatherUnsupportedOp = &op;
                }
              }
              for (auto [queue, tensor] : bodyTempQueueTensors)
                b.create<TQueBindFreeTensorOp>(forLoc, queue, tensor);
            }

            Value zeroVal;
            if (elemType.isF16())
              zeroVal = b.create<arith::ConstantOp>(
                  forLoc, b.getF16FloatAttr(0.0f));
            else if (elemType.isF32())
              zeroVal = b.create<arith::ConstantOp>(
                  forLoc, b.getF32FloatAttr(0.0f));
            if (zeroVal) {
              Value zeroLt =
                  cc.allocVeccalc(b, forLoc, elemType, SmallVector<Value>{dimK})
                      .second;
              auto dupOp =
                  b.create<DuplicateL2Op>(forLoc, zeroLt, zeroVal, dimK_i32);
              copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
              auto copyOp = b.create<AddL2Op>(
                  forLoc, dstRowLt, gatheredRowLt, zeroLt, dimK_i32);
              copyAscendCUnitAttr(genOp.getOperation(), copyOp.getOperation());
            }

            b.create<TQueBindFreeTensorOp>(forLoc, dataRowQueue, dataRowLt);
            b.create<scf::YieldOp>(forLoc);
          });

      if (gatherUnsupportedOp) {
        genOp.emitError("LinalgToAscendC: unsupported op in gather pre/post/"
                        "fused body: ")
            << gatherUnsupportedOp->getName();
        return failure();
      }

      if (Value q = ctx.getQueue(outMemref))
        builder.create<TQueBindEnqueTensorOp>(loc, q, dstLt);

      if (postOp) postOp.erase();
      if (preOp) preOp.erase();
      genOp.erase();
      continue;
    }

    // ---- Transpose generic: emit ascendc.transpose ----
    if (isTransposeGeneric(genOp)) {
      // Only the rank-2 [1,0] 16-bit case is correct via AscendC::Transpose;
      // refuse anything else rather than silently emit a wrong vtranspose.
      {
        AffineMap inMap = genOp.getIndexingMapsArray()[0];
        SmallVector<int64_t> perm;
        for (AffineExpr e : inMap.getResults())
          perm.push_back(
              static_cast<int64_t>(cast<AffineDimExpr>(e).getPosition()));
        auto inMrt =
            cast<MemRefType>(genOp.getDpsInputOperand(0)->get().getType());
        if (!transposeSupportedByIntrinsic(inMrt.getElementType(), perm)) {
          genOp.emitError(
              "LinalgToAscendC: AscendC::Transpose only supports a rank-2 "
              "[1,0] transpose of 16-bit data; route this transpose to aclnn");
          return failure();
        }
      }
      Value inMemref = genOp.getDpsInputOperand(0)->get();
      Location loc = genOp.getLoc();
      builder.setInsertionPoint(genOp);
      Type tElemType = cast<MemRefType>(outMemref.getType()).getElementType();

      // --- source tile ---
      // Preserve template: the transpose's on-chip output makes
      // InsertTileBuffers skip the transpose op, so it does NOT pre-load the
      // input — it stays a (row-strided) GM subview that we copy into VECIN
      // here.  Degenerate case (transpose output is a GM target, e.g. a kernel
      // result): InsertTileBuffers did create a VECIN tile, so just readTensor.
      Value srcLt;
      Value srcFreeQueue; // non-null ⇒ FreeTensor srcLt after the transpose
      if (getMemorySpace(inMemref.getType()) == 0 /*GM*/) {
        auto inMrt = cast<MemRefType>(inMemref.getType());
        SmallVector<int64_t> strides;
        int64_t off = 0;
        bool haveStrides = succeeded(inMrt.getStridesAndOffset(strides, off));
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(tElemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        bool rowStrided = haveStrides && inMrt.getRank() == 2 &&
                          strides[1] == 1 && !ShapedType::isDynamic(strides[0]) &&
                          (ShapedType::isDynamic(inMrt.getDimSize(1)) ||
                           inMrt.getDimSize(1) != strides[0]);
        if (rowStrided) {
          Value rows = cc.getDynDim(builder, loc, inMemref, 0);
          Value cols = cc.getDynDim(builder, loc, inMemref, 1);
          Value rowStride =
              builder.create<arith::ConstantIndexOp>(loc, strides[0]);
          auto [deq, q] = cc.copyGmToVecinStrided(builder, builder, loc,
                                                tElemType, srcGt, rows, cols,
                                                rowStride);
          srcLt = deq;
          srcFreeQueue = q;
        } else {
          Value n = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (unsigned d = 0; d < inMrt.getRank(); ++d)
            n = builder.create<arith::MulIOp>(
                loc, n, cc.getDynDim(builder, loc, inMemref, d));
          srcLt = cc.copyGmToVecin(builder, loc, tElemType, srcGt, n);
        }
      } else {
        srcLt = cc.readTensor(builder, loc, inMemref);
      }

      // --- destination tile + transpose ---
      if (getMemorySpace(outMemref.getType()) == 11 /*VECCALC*/) {
        // Preserve template: an on-chip intermediate, possibly read by several
        // consumers — fresh VECCALC TBuf, registered as the live tensor so the
        // consumers reuse it (a depth-1 queue would dead-lock the 2nd reader).
        SmallVector<Value> outDims;
        for (unsigned d = 0;
             d < cast<MemRefType>(outMemref.getType()).getRank(); ++d)
          outDims.push_back(cc.getDynDim(builder, loc, outMemref, d));
        auto [transpTbuf, dstLt] =
            cc.allocVeccalc(builder, loc, tElemType, outDims);
        auto transposeOp = builder.create<TransposeOp>(loc, dstLt, srcLt);
        copyAscendCUnitAttr(genOp.getOperation(), transposeOp.getOperation());
        ctx.setLiveTensor(outMemref, dstLt);
        if (srcFreeQueue)
          builder.create<TQueBindFreeTensorOp>(loc, srcFreeQueue, srcLt);
      } else {
        Value dstLt = cc.writeTensor(builder, loc, outMemref);
        auto transposeOp = builder.create<TransposeOp>(loc, dstLt, srcLt);
        copyAscendCUnitAttr(genOp.getOperation(), transposeOp.getOperation());
        if (Value q = ctx.getQueue(outMemref))
          builder.create<TQueBindEnqueTensorOp>(loc, q, dstLt);
        if (srcFreeQueue)
          builder.create<TQueBindFreeTensorOp>(loc, srcFreeQueue, srcLt);
      }

      genOp.erase();
      continue;
    }

    unsigned numInputs = genOp.getNumDpsInputs();
    auto iterTypes     = genOp.getIteratorTypesArray();
    unsigned iterRank  = iterTypes.size();
    auto maps          = genOp.getIndexingMapsArray();

    Location loc = genOp.getLoc();
    builder.setInsertionPoint(genOp);
    Type elemType = cast<MemRefType>(outMemref.getType()).getElementType();

    // ---- Compute iteration dim sizes from the first full-rank input ----
    SmallVector<Value> iterDimSizes(iterRank);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      bool allDimExprs = llvm::all_of(inMap.getResults(),
          [](AffineExpr e) { return isa<AffineDimExpr>(e); });
      if (inMap.getNumResults() == iterRank && allDimExprs) {
        for (unsigned d = 0; d < iterRank; ++d)
          iterDimSizes[d] = cc.getDynDim(builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining dims from output (all parallel, same rank).
    for (unsigned d = 0; d < iterRank; ++d)
      if (!iterDimSizes[d])
        iterDimSizes[d] = cc.getDynDim(builder, loc, outMemref, d);

    // totalElems is the actual element count for this tile.  Buffer
    // allocation uses the enclosing loop-step upper bound so tail iterations
    // reuse one max-sized queue/tbuf instead of repeatedly InitBuffer-ing.
    Value totalElems = cc.computeProduct(builder, loc, iterDimSizes);
    SmallVector<Value> bufferDimSizes =
        cc.getBufferDimSizes(iterDimSizes, genOp.getOperation());
    Value bufferTotalElems = cc.computeProduct(builder, loc, bufferDimSizes);
    SmallVector<std::pair<Value, Value>> tempVecinTensors;

    Value outQueue = ctx.getQueue(outMemref);
    Value accumLt;
    if (!outQueue) {
      // Allocate the shared VECCALC accumulator for intermediate results.
      auto [accumTbuf, veccalcAccumLt] =
          cc.allocVeccalc(builder, loc, elemType, bufferDimSizes);
      accumLt = veccalcAccumLt;
    }

    // ---- Step 1: Promote each input to a VECCALC local_tensor ----
    SmallVector<Value> inputLts(numInputs);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      int64_t inMs    = getMemorySpace(inMemref.getType());
      IndexingMapAnalysis analysis = analyzeIndexingMap(inMap, iterRank);

      switch (analysis.kind) {
      case IndexingMapAnalysis::Kind::Identity: {
        if (inMs == 0 /*GM*/) {
          // GM input at full rank: copy via VECIN TQue.
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          inputLts[i] =
              cc.copyGmToVecin(builder, loc, elemType, srcGt, totalElems,
                            bufferTotalElems, &tempVecinTensors);
        } else {
          inputLts[i] = cc.readTensor(builder, loc, inMemref);
        }
        break;
      }
      case IndexingMapAnalysis::Kind::PureBroadcast: {
        auto srcMrt = cast<MemRefType>(inMemref.getType());
        unsigned srcRank = srcMrt.getRank();
        if (inMs == 9 /*VECIN*/ || inMs == 11 /*VECCALC*/) {
          // broadcast_l2: expand narrow VECIN/VECCALC tile into full-shape
          // VECCALC.  The VECCALC case is an intra-group intermediate (e.g. the
          // result s[a] of a sibling reduce generic, registered as a live
          // tensor) being broadcast back over the reduced axis — readTensor
          // resolves it via getLiveTensor.
          SmallVector<Value> dstShapeVals, srcShapeVals;
          for (Value s : iterDimSizes)
            dstShapeVals.push_back(
                builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), s));
          unsigned srcDimIdx = 0;
          for (unsigned d = 0; d < iterRank; ++d) {
            bool inResult = false;
            for (AffineExpr result : inMap.getResults())
              if (auto dimExpr = dyn_cast<AffineDimExpr>(result))
                if (dimExpr.getPosition() == d) { inResult = true; break; }
            if (inResult && srcDimIdx < srcRank)
              srcShapeVals.push_back(builder.create<arith::IndexCastOp>(
                  loc, builder.getI32Type(),
                  cc.getDynDim(builder, loc, inMemref, srcDimIdx++)));
            else
              srcShapeVals.push_back(
                  builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
          }
          Value srcLt = cc.readTensor(builder, loc, inMemref);
          auto [bcastTbuf, bcastLt] =
              cc.allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
          inputLts[i] = bcastLt;
        } else {
          // broadcast from GM: copy via VECIN TQue first, then broadcast_l2.
          SmallVector<Value> srcDims;
          for (unsigned d = 0; d < srcRank; ++d)
            srcDims.push_back(cc.getDynDim(builder, loc, inMemref, d));
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDims)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          Value srcLt =
              cc.copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                            srcElemCount, &tempVecinTensors);
          SmallVector<Value> dstShapeVals, srcShapeVals;
          for (Value s : iterDimSizes)
            dstShapeVals.push_back(
                builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), s));
          unsigned srcDimIdx = 0;
          for (unsigned d = 0; d < iterRank; ++d) {
            bool inResult = false;
            for (AffineExpr result : inMap.getResults())
              if (auto dimExpr = dyn_cast<AffineDimExpr>(result))
                if (dimExpr.getPosition() == d) { inResult = true; break; }
            if (inResult && srcDimIdx < srcRank)
              srcShapeVals.push_back(builder.create<arith::IndexCastOp>(
                  loc, builder.getI32Type(), srcDims[srcDimIdx++]));
            else
              srcShapeVals.push_back(
                  builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
          }
          auto [bcastTbuf, bcastLt] =
              cc.allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
          inputLts[i] = bcastLt;
        }
        break;
      }
      case IndexingMapAnalysis::Kind::PureTranspose: {
        // Same intrinsic restriction as the standalone transpose handler: only
        // a rank-2 [1,0] 16-bit swap is correct via vtranspose.
        if (!transposeSupportedByIntrinsic(elemType, analysis.permutation)) {
          genOp.emitError(
              "LinalgToAscendC: AscendC::Transpose only supports a rank-2 "
              "[1,0] transpose of 16-bit data; route this transpose to aclnn");
          return failure();
        }
        // The transpose source may already be on-chip (VECIN) — e.g. when the
        // transpose was absorbed into a downstream elementwise generic and
        // InsertTileBuffers placed a VECIN tile for the operand (whose load was
        // emitted as a memref.copy and lowered by convertDataMove, registering
        // the live tensor).  In that case reuse it; otherwise copy from GM.
        // Mirrors the BroadcastTranspose case below.
        Value srcVecinLt;
        if (inMs == 9 /*VECIN*/) {
          srcVecinLt = cc.readTensor(builder, loc, inMemref);
        } else {
          SmallVector<Value> srcDims;
          for (int64_t permDim : analysis.permutation)
            srcDims.push_back(iterDimSizes[static_cast<unsigned>(permDim)]);
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDims)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);

          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          srcVecinLt =
              cc.copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount);
        }

        auto [transpTbuf, transpLt] =
            cc.allocVeccalc(builder, loc, elemType, bufferDimSizes);
        auto transposeOp = builder.create<TransposeOp>(loc, transpLt, srcVecinLt);
        copyAscendCUnitAttr(genOp.getOperation(), transposeOp.getOperation());
        inputLts[i] = transpLt;
        break;
      }
      case IndexingMapAnalysis::Kind::BroadcastTranspose: {
        // Step 1: Get the input as a local_tensor in VECIN.
        // After buffer-placement, data0 may already be in VECIN (inMs==9) via
        // a memref.copy placeholder; use readTensor directly. Otherwise copy
        // from GM.
        auto srcMrt = cast<MemRefType>(inMemref.getType());
        unsigned srcRank = srcMrt.getRank();
        SmallVector<Value> srcDimsVals;
        for (unsigned d = 0; d < srcRank; ++d)
          srcDimsVals.push_back(cc.getDynDim(builder, loc, inMemref, d));

        Value srcVecinLt;
        if (inMs == 9 /*VECIN*/) {
          srcVecinLt = cc.readTensor(builder, loc, inMemref);
        } else {
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDimsVals)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          srcVecinLt =
              cc.copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                            srcElemCount, &tempVecinTensors);
        }

        // Step 2: Broadcast directly into iteration-space order [iterDimSizes]
        // without a Transpose. AscendC::Transpose(dst, src) only works
        // correctly for square matrices; non-square cases produce wrong results
        // in the simulator. Instead, build dstShape = iterDimSizes and srcShape
        // with broadcast dims set to 1 and present dims set to their sizes.
        //
        // Example: map (d0,d1)->(d1,0), iterDimSizes=[Tb_N, M], src=[M,1]
        //   dstShape = [Tb_N, M]
        //   srcShape = [1,    M]   (d0=broadcast→1, d1=present→M)
        //   axis=0 (first src dim is 1, i.e. broadcast along first axis)
        unsigned iterRank = iterDimSizes.size();
        SmallVector<Value> bcastDstShape, bcastSrcShape;
        for (unsigned d = 0; d < iterRank; ++d) {
          bcastDstShape.push_back(builder.create<arith::IndexCastOp>(
              loc, builder.getI32Type(), iterDimSizes[d]));
        }
        // srcShape: for each iteration dim, if it appears in presentDims of the
        // map put the actual src size, otherwise put 1 (broadcast dim).
        // We need to map iter-dim → src-dim via the src's indexing map results.
        // Build a lookup: iter dim position → src dim index (or -1 if broadcast).
        SmallVector<int64_t> iterDimToSrcDim(iterRank, -1);
        for (unsigned r = 0; r < inMap.getNumResults(); ++r) {
          AffineExpr expr = inMap.getResult(r);
          if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
            // iter dim dimExpr.getPosition() maps to src dimension r
            iterDimToSrcDim[dimExpr.getPosition()] = static_cast<int64_t>(r);
          }
        }
        for (unsigned d = 0; d < iterRank; ++d) {
          int64_t srcDimIdx = iterDimToSrcDim[d];
          if (srcDimIdx >= 0) {
            bcastSrcShape.push_back(builder.create<arith::IndexCastOp>(
                loc, builder.getI32Type(), srcDimsVals[srcDimIdx]));
          } else {
            bcastSrcShape.push_back(
                builder.create<arith::ConstantOp>(
                    loc, builder.getI32IntegerAttr(1)));
          }
        }

        auto [finalTbuf, finalLt] =
            cc.allocVeccalc(builder, loc, elemType, bufferDimSizes);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, finalLt, srcVecinLt,
            bcastDstShape, bcastSrcShape,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = finalLt;
        break;
      }
      } // end switch
    }

    // The body-walk writes every op result into a single in-place accumulator
    // (accumLt).  That is only safe for a linear chain where each value is read
    // exactly once: if any value (a block-arg input OR an intermediate result)
    // is read by more than one op, the in-place overwrite clobbers a value that
    // is still needed later (e.g. fused bias-add+GELU reads x = bias-add result
    // in both x/sqrt2 and the final *x → x*x instead of x*0.5(1+erf)).  For such
    // "DAG" bodies, give each op its own fresh VECCALC dst so no value is
    // overwritten while live.  Linear bodies keep the cheap in-place accumulator.
    Block &bodyBlock = *genOp.getBody();
    bool dagBody = false;
    {
      llvm::SmallDenseMap<Value, unsigned> useCount;
      for (Operation &bodyOp : bodyBlock.without_terminator())
        for (Value operand : bodyOp.getOperands()) {
          bool counts = false;
          if (auto ba = dyn_cast<BlockArgument>(operand))
            counts = ba.getArgNumber() < numInputs;
          else if (operand.getDefiningOp() &&
                   operand.getDefiningOp()->getBlock() == &bodyBlock)
            counts = true; // intermediate body result
          if (counts && ++useCount[operand] > 1)
            dagBody = true;
        }
    }

    if (outQueue && !dagBody)
      accumLt = cc.allocTensor(builder, loc, outQueue, elemType);
    else if (outQueue)
      accumLt = cc.allocVeccalc(builder, loc, elemType, bufferDimSizes).second;

    // ---- Step 2: Walk body and inline arith ops onto VECCALC tensors ----
    // Per-op dst: a fresh VECCALC for DAG bodies, else the in-place accumulator.
    auto nextDst = [&]() -> Value {
      return dagBody ? cc.allocVeccalc(builder, loc, elemType, bufferDimSizes).second
                     : accumLt;
    };
    unsigned numBodyArgs = bodyBlock.getNumArguments();
    SmallVector<Value> argToLt(numBodyArgs);
    for (unsigned i = 0; i < numInputs; ++i)
      argToLt[i] = inputLts[i];
    argToLt[numInputs] = accumLt;

    llvm::SmallDenseMap<Value, Value> valToLt;
    // Guard body entry against MTE2 → V race on real-NPU.  Multi-trial
    // statistics on BERT group20 (HEAD with Erf/EnQue PIPE_V barriers)
    // still showed ~20% sub-deterministic failures, zero_frac concentrated
    // at row-0 cols-0/1 — i.e. the FIRST vector op of a body reads from
    // a VECIN tensor before MTE2's last cycle commits.  Sim serializes
    // loads so it never trips.  A pre-body PIPE_ALL barrier flushes any
    // pending MTE2/MTE3 before the first compute op runs.
    //
    // Mix-kernel guard: CannTranslation's mix-kernel emitter does its own
    // strict single-executable-chain analysis on the AIV partition and
    // rejects PipeBarrier ops "outside the chain" (matmul-add-leakyrelu
    // mix kernel breaks otherwise).  Mix kernels have their own pipeline
    // management, so skip both this barrier and the pre-EnQue barrier.
    bool isMixKernel = false;
    if (auto kk = funcOp->getAttrOfType<StringAttr>("ascendc.kernel_kind"))
      isMixKernel = kk.getValue() == "mix";
    if (!isMixKernel) {
      auto bodyStartBarrier = builder.create<ascendc::PipeBarrierOp>(
          loc, ascendc::PipeAttr::get(builder.getContext(),
                                       ascendc::Pipe::PIPE_ALL));
      copyAscendCUnitAttr(genOp.getOperation(),
                          bodyStartBarrier.getOperation());
    }
    for (auto &bodyOp : bodyBlock.without_terminator()) {
      auto resolve = [&](Value v) -> Value {
        if (auto ba = dyn_cast<BlockArgument>(v))
          return argToLt[ba.getArgNumber()];
        auto it = valToLt.find(v);
        if (it != valToLt.end()) return it->second;
        // Scalar constant? Fill a fresh VECCALC with duplicate_l2.
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          auto [dupTbuf, dupLt] =
              cc.allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
          valToLt[v] = dupLt;
          return dupLt;
        }
        return Value{};
      };

      if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
        Value lhs = resolve(addOp.getLhs());
        Value rhs = resolve(addOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = nextDst();
        auto addL2Op = builder.create<AddL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), addL2Op.getOperation());
        valToLt[addOp.getResult()] = dst;
      } else if (auto subOp = dyn_cast<arith::SubFOp>(bodyOp)) {
        Value lhs = resolve(subOp.getLhs());
        Value rhs = resolve(subOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = nextDst();
        auto subL2Op = builder.create<SubL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), subL2Op.getOperation());
        valToLt[subOp.getResult()] = dst;
      } else if (auto mulOp = dyn_cast<arith::MulFOp>(bodyOp)) {
        Value lhs = resolve(mulOp.getLhs());
        Value rhs = resolve(mulOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = nextDst();
        auto mulL2Op = builder.create<MulL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), mulL2Op.getOperation());
        valToLt[mulOp.getResult()] = dst;
      } else if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
        Value lhs = resolve(maxOp.getLhs());
        Value rhs = resolve(maxOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = nextDst();
        auto maxL2Op = builder.create<MaxL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), maxL2Op.getOperation());
        valToLt[maxOp.getResult()] = dst;
      } else if (auto minOp = dyn_cast<arith::MinimumFOp>(bodyOp)) {
        Value lhs = resolve(minOp.getLhs());
        Value rhs = resolve(minOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = nextDst();
        auto minL2Op = builder.create<MinL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), minL2Op.getOperation());
        valToLt[minOp.getResult()] = dst;
      } else if (auto divOp = dyn_cast<arith::DivFOp>(bodyOp)) {
        Value lhs = resolve(divOp.getLhs());
        Value rhs = resolve(divOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = nextDst();
        auto divL2Op = builder.create<DivL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), divL2Op.getOperation());
        valToLt[divOp.getResult()] = dst;
      } else if (auto erfOp = dyn_cast<math::ErfOp>(bodyOp)) {
        // AscendC::Erf forbids src/dst overlap, so it cannot reuse the in-place
        // accumLt — emit into a fresh VECCALC tensor. Uses the simple overload
        // (no caller tmp buffer); the math advanced-API manages its own scratch.
        //
        // PIPE_V barrier follows: AscendC::Erf is a deep-pipeline math
        // advanced-API; its result isn't committed by the time chained vector
        // ops would naively read it.  Sim serializes ops so the data race is
        // invisible, but real 910C with Erf in the middle of a body (BERT
        // GELU group20) observed downstream VECCALC writes / VECOUT EnQue
        // capturing partial results — symptom = 1-row-per-block (12.5% on
        // `bd=4 8×256` shape) of the OUTPUT GM stayed at host-side zero-init.
        // The barrier serializes V-pipe so subsequent ops wait for Erf.
        Value src = resolve(erfOp.getOperand());
        if (!src) continue;
        Value erfDst = cc.allocVeccalc(builder, loc, elemType, bufferDimSizes).second;
        std::string ets = cppScalarName(elemType);
        // PIPE_V (not PIPE_ALL): empirically PIPE_V drives BERT group20
        // zero_frac 12.5%→3.1%; PIPE_ALL regressed to max_diff=0.4
        // (over-serialization breaks intended V/MTE overlap).  PIPE_V
        // serializes consecutive V-pipe consumers, which is what Erf's
        // deep latency needs.  The residual ~3% miss / max_diff~1.7 needs
        // a follow-up — barrier before VECOUT EnQue or finer-grained
        // serialization between Erf-output consumers, TBD.
        std::string tmpl = "AscendC::Erf<" + ets +
                           ", false>($0, $1, (uint32_t)$2);\n"
                           "  AscendC::PipeBarrier<PIPE_V>()";
        auto vb = builder.create<emitasc::VerbatimOp>(
            loc, builder.getStringAttr(tmpl), ValueRange({erfDst, src, totalElems}));
        copyAscendCUnitAttr(genOp.getOperation(), vb.getOperation());
        valToLt[erfOp.getResult()] = erfDst;
      } else if (!isa<arith::ConstantOp>(bodyOp)) {
        // Fail loudly rather than silently emitting a kernel that drops this op.
        genOp.emitError("LinalgToAscendC: unsupported op in linalg.generic "
                        "body: ")
            << bodyOp.getName();
        return failure();
      }
    }

    // ---- Step 3: Write accumulator to output buffer ----
    // No reduction needed (all-parallel). The compute result is in accumLt
    // (a VECCALC tbuf). We need to deliver it to the output buffer:
    //
    //   VECOUT (ms=10): alloc from queue, use AddL2 to copy accumLt→vecoutLt
    //                   (add_l2(dst, src, zero_tbuf, count) would need a zero
    //                    tensor; instead use the queue alloc tensor directly and
    //                    simply enqueue accumLt if the queue accepts VECCALC).
    //                   Simplest: treat the VECCALC accumLt as the enqueue source
    //                   and let the downstream DataMoveConversion handle writeback.
    //   VECCALC (ms=11): accumLt already holds the result; no copy needed.
    //
    // Key insight: the epilogue memref.copy (VECOUT→GM) inserted by
    // AscendCBufferPlacement is converted by DataMoveConversion into a
    // data_copy_l2 with the correct subview offset, so the Concat position
    // is preserved automatically. We just need to enqueue the result tensor.
    // Resolve the yielded value to its tensor.  A pure copy / broadcast body
    // yields a block argument (an input) directly: the result lives in that
    // input's promoted tensor (e.g. the BroadcastL2 destination), NOT in
    // accumLt — the empty body never wrote accumLt, so enqueueing accumLt would
    // emit an all-zero output (the leading-axis weight-broadcast bug).  A
    // compute body yields a value in valToLt (== accumLt), so this is a no-op
    // for that case.
    Value resultLt = accumLt;
    if (auto yieldOp = dyn_cast<linalg::YieldOp>(bodyBlock.getTerminator())) {
      if (yieldOp.getNumOperands() == 1) {
        Value yielded = yieldOp.getOperand(0);
        if (auto ba = dyn_cast<BlockArgument>(yielded)) {
          if (ba.getArgNumber() < numInputs)
            resultLt = inputLts[ba.getArgNumber()];
        } else if (auto it = valToLt.find(yielded); it != valToLt.end()) {
          resultLt = it->second;
        }
      }
    }

    if (outQueue) {
      // The queue expects a tensor allocated from the same queue.  Real
      // hardware is stricter than the simulator here; enqueueing a VECCALC
      // tbuf tensor into a VECOUT queue can surface as UB/MTE faults.
      //
      // PIPE_V barrier before EnQue: a body that chains many V-pipe ops
      // (e.g. BERT GELU group20: bias-add → divf → erf → addf → mulf →
      // mulf, writing into resultLt at the tail) hits a real-NPU race
      // where the VECOUT consumer DMA reads resultLt's first 1-2
      // elements before the last V-pipe write commits (~50% trial rate,
      // row-0-cols-0/1 stay at host zero-init).  Force a V-pipe drain
      // so EnQue's MTE3-trigger observes the full computed tensor.  Sim
      // serializes V naturally and never trips this; only real HW races.
      //
      // Mix-kernel guard: same reason as the body-start barrier — mix
      // kernel emitter rejects extra ops in the single executable chain.
      if (!isMixKernel) {
        auto preEnqueBarrier = builder.create<ascendc::PipeBarrierOp>(
            loc, ascendc::PipeAttr::get(builder.getContext(),
                                         ascendc::Pipe::PIPE_V));
        copyAscendCUnitAttr(genOp.getOperation(),
                            preEnqueBarrier.getOperation());
      }
      builder.create<TQueBindEnqueTensorOp>(loc, outQueue, resultLt);
    }
    cc.freeTempVecinTensors(builder, loc, tempVecinTensors);
    // If outMemref has no queue (VECCALC alloc without a queue), the result
    // already resides in the VECCALC tbuf and will be consumed by the next op.

    genOp.erase();
  }

  return success();
}

} // namespace afir
} // namespace mlir
