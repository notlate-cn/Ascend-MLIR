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

#include "Conversion/LinalgToAscendC/ComputeConversionHelpers.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace afir {

void copyAscendCUnitAttr(Operation *src, Operation *dst) {
  if (!src || !dst)
    return;
  if (auto unitAttr = src->getAttrOfType<StringAttr>("ascendc.unit"))
    dst->setAttr("ascendc.unit", unitAttr);
}

Value getEnclosingLoopStepBound(Value value, Operation *anchor) {
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

scf::ForOp getEnclosingFor(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p))
      return f;
  return nullptr;
}

IndexingMapAnalysis analyzeIndexingMap(AffineMap map, unsigned iterRank) {
  IndexingMapAnalysis result;

  // Identity: fast path
  if (map.isIdentity()) {
    result.kind = IndexingMapAnalysis::Kind::Identity;
    return result;
  }

  // Collect which iteration dims appear in the map results (as dim exprs)
  // and which results are constants.
  SmallVector<int64_t> presentDims;  // iteration dim positions that appear
  bool hasConstant = false;
  for (AffineExpr expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      presentDims.push_back(static_cast<int64_t>(dimExpr.getPosition()));
    } else if (isa<AffineConstantExpr>(expr)) {
      hasConstant = true;
    } else {
      // Non-trivial affine expression: not handled.
      result.kind = IndexingMapAnalysis::Kind::Identity; // fallback: treat as identity
      return result;
    }
  }

  // Determine broadcast dims: iteration dims not in presentDims.
  for (unsigned d = 0; d < iterRank; ++d) {
    if (llvm::find(presentDims, static_cast<int64_t>(d)) == presentDims.end())
      result.broadcastDims.push_back(d);
  }

  bool hasBroadcast = !result.broadcastDims.empty() || hasConstant;
  bool hasTranspose = !llvm::is_sorted(presentDims);

  if (hasConstant || (hasBroadcast && hasTranspose)) {
    result.kind = IndexingMapAnalysis::Kind::BroadcastTranspose;
    result.permutation.assign(presentDims.begin(), presentDims.end());
    return result;
  }

  if (hasBroadcast) {
    result.kind = IndexingMapAnalysis::Kind::PureBroadcast;
    return result;
  }

  if (hasTranspose) {
    result.kind = IndexingMapAnalysis::Kind::PureTranspose;
    result.permutation.assign(presentDims.begin(), presentDims.end());
    return result;
  }

  result.kind = IndexingMapAnalysis::Kind::Identity;
  return result;
}

bool isBroadcastMap(AffineMap map, unsigned iterRank) {
  if (map.getNumResults() >= iterRank)
    return false;
  return true;
}

std::string cppScalarName(Type t) {
  if (t.isF32()) return "float";
  if (t.isF16()) return "half";
  if (t.isBF16()) return "bfloat16_t";
  if (auto it = dyn_cast<IntegerType>(t))
    return "int" + std::to_string(it.getWidth()) + "_t";
  return "float";
}

bool isTransposeGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return false;
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 2)
    return false;
  AffineMap inMap  = maps[0];
  AffineMap outMap = maps[1];
  unsigned rank    = op.getIteratorTypesArray().size();
  if (rank == 0)
    return false;
  // Output must be identity
  if (!outMap.isIdentity())
    return false;
  // Input must have same rank as iteration space (no broadcast)
  if (inMap.getNumResults() != rank)
    return false;
  // All input map results must be distinct AffineDimExprs (no constants, no complex exprs)
  SmallVector<int64_t> perm(rank, -1);
  for (unsigned r = 0; r < rank; ++r) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(r));
    if (!dimExpr)
      return false;
    int64_t pos = static_cast<int64_t>(dimExpr.getPosition());
    if (pos < 0 || pos >= static_cast<int64_t>(rank))
      return false;
    perm[r] = pos;
  }
  // All positions must be distinct (no repeated dim in permutation)
  llvm::SmallDenseSet<int64_t> seen;
  for (unsigned r = 0; r < rank; ++r)
    if (!seen.insert(perm[r]).second)
      return false;
  // Must be a non-identity permutation
  bool isIdentityPerm = true;
  for (unsigned r = 0; r < rank; ++r)
    if (perm[r] != static_cast<int64_t>(r)) { isIdentityPerm = false; break; }
  if (isIdentityPerm)
    return false;
  // Body must be yield-only (single linalg.yield yielding the input block arg)
  Block &body = *op.getBody();
  if (body.getOperations().size() != 1)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body.front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  auto ba = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  return ba && ba.getArgNumber() == 0;
}

bool transposeSupportedByIntrinsic(Type elemType, ArrayRef<int64_t> perm) {
  // vtranspose dtype set is half / int16 / uint16 (bf16 is NOT supported).
  if (!(elemType.isF16() || elemType.isInteger(16)))
    return false;
  // Only a rank-2 [1,0] swap maps to vtranspose's 2D semantics.
  return perm.size() == 2 && perm[0] == 1 && perm[1] == 0;
}

bool transposeSupportedByConfusion(Type elemType, ArrayRef<int64_t> perm) {
  // AF ConfusionTranspose (ND2ND_ONLY) handles f16 and f32 at any [H,W] size;
  // the MVP supports the rank-2 [1,0] swap.
  if (!(elemType.isF16() || elemType.isF32()))
    return false;
  return perm.size() == 2 && perm[0] == 1 && perm[1] == 0;
}

bool isIndexSelectGeneric(linalg::GenericOp op) {
  return op->hasAttr("gather_dim");
}

bool isEmbeddingGeneric(linalg::GenericOp op) {
  return op->hasAttr("embedding_dim");
}

} // namespace afir
} // namespace mlir
