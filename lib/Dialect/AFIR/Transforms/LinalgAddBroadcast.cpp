//===- LinalgAddBroadcast.cpp - Linalg shape inference pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <functional>
#include <numeric>
namespace mlir {

#define GEN_PASS_DEF_LINALGADDBROADCASTPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

std::optional<std::pair<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>>> getBroadcastAndTransposeFromMaps(
    mlir::AffineMap A, mlir::AffineMap B) {
  if (A.getNumDims() != B.getNumDims()) return std::nullopt;

  // 1. 建立从维度位置到 B 结果索引的映射
  llvm::SmallDenseMap<unsigned, unsigned> posToBIndex;
  for (unsigned i = 0, e = B.getNumResults(); i < e; ++i) {
    auto expr = B.getResult(i);
    auto dimExpr = dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr) return std::nullopt;  // B 不是投影置换
    unsigned pos = dimExpr.getPosition();
    // 确保 B 中没有重复维度
    if (posToBIndex.count(pos)) return std::nullopt;
    posToBIndex[pos] = i;
  }

  // 2. 收集 A 中出现的所有维度位置，并构建 currentOrder
  llvm::SmallDenseSet<unsigned> dimsInA;
  llvm::SmallVector<unsigned> currentOrder;
  for (auto expr : A.getResults()) {
    auto dimExpr = dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr) return std::nullopt;  // A 不是投影置换
    unsigned pos = dimExpr.getPosition();
    if (!posToBIndex.count(pos)) return std::nullopt;  // A 中出现了 B 中没有的维度
    dimsInA.insert(pos);
    currentOrder.push_back(posToBIndex[pos]);
  }

  // 3. 检查 currentOrder 是否严格递增（即已自然顺序）
  bool isIdentity = true;
  for (size_t i = 1; i < currentOrder.size(); ++i) {
    if (currentOrder[i] <= currentOrder[i - 1]) {
      isIdentity = false;
      break;
    }
  }

  llvm::SmallVector<int64_t> permutation;
  if (!isIdentity) {
    // 4. 计算置换向量：对 currentOrder 排序得到原始索引的排列
    permutation.resize(currentOrder.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    // 根据 currentOrder 的值对 permutation 进行稳定排序
    std::stable_sort(permutation.begin(), permutation.end(),
                     [&](unsigned i, unsigned j) { return currentOrder[i] < currentOrder[j]; });
    // 此时 permutation 表示目标顺序的第 i 个位置来自原 permutation[i]
    // 这正是 linalg.transpose 所需的置换（将原维度重排到目标顺序）
  }

  // 5. 计算广播维度：B 中哪些结果索引对应的维度不在 dimsInA 中
  llvm::SmallVector<int64_t> broadcastedDims;
  for (unsigned i = 0, e = B.getNumResults(); i < e; ++i) {
    auto expr = B.getResult(i);
    unsigned pos = cast<mlir::AffineDimExpr>(expr).getPosition();
    if (!dimsInA.contains(pos)) broadcastedDims.push_back(i);
  }

  return std::make_pair(permutation, broadcastedDims);
}

std::pair<SmallVector<int64_t>, SmallVector<int64_t>> computeTransposeBroadcast(AffineMap &map) {
  assert(map.isProjectedPermutation(false) && "not a projection");

  // As the map is a projection it likely operates on a smaller set of
  // dimensions as far as the transpose is concerned (rest are broadcast).
  int64_t minorSize = map.getNumResults();

  SmallVector<int64_t> minorResult;
  for (int64_t i = 0; i < minorSize; ++i) {
    auto expr = cast<AffineDimExpr>(map.getResults()[i]);
    minorResult.push_back(expr.getPosition());
  }

  // If dims are not monotonically increasing then transpose is present.
  SmallVector<int64_t> sortedResMap(minorResult);
  llvm::sort(sortedResMap);
  bool hasTranspose = !std::equal(minorResult.begin(), minorResult.end(), sortedResMap.begin(), sortedResMap.end());

  // Walk the sorted map result to determine which dimensions are broadcasted.
  SmallVector<int64_t> broadcast;
  for (int64_t i = 0, j = 0; i < map.getNumInputs(); ++i) {
    if (j < minorSize && sortedResMap[j] == i) {
      j++;
      continue;
    }
    broadcast.push_back(i);
  }

  SmallVector<int64_t> permutation;
  if (hasTranspose) {
    // Consider an operand `x : tensor<7x8x9>` of a genericOp that has
    // affine map `affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>`
    // `x`s access is both transposed and broadcast. But when specifying
    // the `linalg.transpose(x : tensor<7x8x9>)` the dimensions need to be
    // specified as `affine_map<(d0,d1,d2) -> (d1, d2, d0)` instead of
    // refering to d3, d4. Therefore, re-base the transpose dimensions so
    // that they start from d0.
    permutation.resize(minorSize);
    std::map<int64_t, int64_t> minorMap;
    for (int64_t i = 0; i < minorSize; ++i) minorMap.insert({sortedResMap[i], i});

    // Re-map the dimensions.
    SmallVector<int64_t> remappedResult(minorSize);
    for (int64_t i = 0; i < minorSize; ++i) remappedResult[i] = minorMap[minorResult[i]];

    /// Calculate the permutation for the transpose.
    for (int64_t i = 0; i < minorSize; ++i) {
      permutation[remappedResult[i]] = i;
    }
  }
  return {permutation, broadcast};
}

void makeCollapseShape(linalg::LinalgOp op, SmallVector<AffineMap> &newIndexingMap, PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  for (auto [idx, oper] : llvm::enumerate(op.getDpsInputOperands())) {
    AffineMap map = newIndexingMap[idx];
    SmallVector<unsigned> broadcastDims(map.getBroadcastDims());
    if (broadcastDims.size()) {
      AffineMap newMap(map);
      SmallVector<ReassociationIndices> reassociation;
      auto rank = cast<RankedTensorType>(oper->get().getType()).getRank();
      for (int i = 0, j = 0; i < rank; i++) {
        if (j == broadcastDims.size()) {
          reassociation.push_back({i});
        } else {
          if (i < broadcastDims[j]) {
            reassociation.push_back({i});
          } else {
            j++;
          }
        }
      }
      reassociation.insert(reassociation.begin(), {-1});
      reassociation.insert(reassociation.end(), {rank + 2});
      for (int i = 0, j = 0; i < broadcastDims.size(); i++) {
        while (j + 1 < reassociation.size()) {
          if (broadcastDims[i] > reassociation[j].back() && broadcastDims[i] < reassociation[j + 1].back()) {
            break;
          }
          j++;
        }
        if (j == 0) {
          reassociation[j + 1].insert(reassociation[j + 1].begin() + (reassociation[j + 1].size() - 1),
                                      broadcastDims[i]);
        } else {
          reassociation[j].push_back(broadcastDims[i]);
        }
      }
      newMap = newMap.dropZeroResults();
      reassociation.pop_back();
      reassociation.erase(reassociation.begin());
      Value newOper = rewriter.create<tensor::CollapseShapeOp>(loc, oper->get(), reassociation);
      rewriter.modifyOpInPlace(op, [&]() { op->setOperand(idx, newOper); });
      newIndexingMap[idx] = newMap;
    }
  }
  if (op.getIndexingMapsArray() != newIndexingMap) {
    op->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(newIndexingMap));
  }
}

LogicalResult elementwiseBroadcast(linalg::LinalgOp op, PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  SmallVector<AffineMap> newIndexingMap(op.getIndexingMapsArray());
  makeCollapseShape(op, newIndexingMap, rewriter);
  IRMapping mapper;
  for (auto [idx, oper] : llvm::enumerate(op.getDpsInputOperands())) {
    auto map = newIndexingMap[idx];
    auto x = getBroadcastAndTransposeFromMaps(map, rewriter.getMultiDimIdentityMap(map.getNumDims()));
    if (x.has_value()) {
      auto [permutationDim, broadcastDim] = x.value();
      if (permutationDim.size() || broadcastDim.size()) {
        auto x = op.createLoopRanges(rewriter, loc);
        SmallVector<Value> dims;
        for (int i = 0; i < op.getNumLoops(); i++) {
          if (isa<Value>(x[i].size)) {
            dims.push_back(dyn_cast<Value>(x[i].size));
          }
        }
        Value broadcasted;
        if (op.hasPureTensorSemantics()) {
          broadcasted = rewriter.create<tensor::EmptyOp>(
              loc,
              TypeRange{RankedTensorType::get(op.getStaticLoopRanges(),
                                              cast<ShapedType>(oper->get().getType()).getElementType())},
              dims);
        } else {
          broadcasted = rewriter.create<memref::AllocOp>(
              loc, MemRefType::get(op.getStaticLoopRanges(), cast<ShapedType>(oper->get().getType()).getElementType()),
              dims);
        }
        Value currentValue = oper->get();
        if (permutationDim.size()) {
          SmallVector<int64_t> originShape(cast<ShapedType>(oper->get().getType()).getShape());
          applyPermutationToVector(originShape, permutationDim);
          SmallVector<Value> dynDims;
          for (auto [idx, s] : llvm::enumerate(cast<ShapedType>(oper->get().getType()).getShape())) {
            if (ShapedType::isDynamic(s)) {
              Value dim = op.hasPureTensorSemantics()
                              ? rewriter.create<tensor::DimOp>(loc, currentValue, idx).getResult()
                              : rewriter.create<memref::DimOp>(loc, currentValue, idx).getResult();
              dynDims.push_back(dim);
            }
          }
          Value emptyValue;
          if (op.hasPureTensorSemantics()) {
            emptyValue = rewriter.create<tensor::EmptyOp>(
                loc,
                TypeRange{RankedTensorType::get(originShape, cast<ShapedType>(oper->get().getType()).getElementType())},
                dynDims);
          } else {
            emptyValue = rewriter.create<memref::AllocOp>(
                loc, MemRefType::get(originShape, cast<ShapedType>(oper->get().getType()).getElementType()), dynDims);
          }
          auto transpose = rewriter.create<linalg::TransposeOp>(loc, currentValue, emptyValue, permutationDim);
          if (op.hasPureTensorSemantics()) {
            currentValue = transpose.getResult()[0];
          } else {
            currentValue = emptyValue;
          }
        }
        auto broadcastOp = rewriter.create<linalg::BroadcastOp>(loc, currentValue, broadcasted, broadcastDim);
        mapper.map(oper->get(), op.hasPureTensorSemantics() ? broadcastOp.getResult()[0] : broadcasted);
        newIndexingMap[idx] = AffineMap::getMultiDimIdentityMap(op.getNumLoops(), op.getContext());
      }
    }
  }
  if (mapper.getValueMap().size()) {
    auto newOp = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, newOp);
    auto newIndexingMaps = rewriter.getAffineMapArrayAttr(newIndexingMap);
    newOp->setAttr("indexing_maps", newIndexingMaps);
  }
  return success();
}

struct LinalgBroadcast : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op, PatternRewriter &rewriter) const override {
    auto attr = op->getAttr("namedKind");
    if (attr == Attribute() || !isa<IntegerAttr>(attr)) {
      return failure();
    }
    auto integerAttr = dyn_cast<IntegerAttr>(attr);
    switch (integerAttr.getInt()) {
      case 2: {
        elementwiseBroadcast(op, rewriter);
        return success();
      }
    }
    return failure();
  }
};

struct LinalgTotalBroadcast : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op, PatternRewriter &rewriter) const override {
    IRMapping mapper;
    SmallVector<AffineMap> newAttr(op.getIndexingMapsArray());
    for (auto [idx, oper] : llvm::enumerate(op.getDpsInputOperands())) {
      auto map = op.getIndexingMapsArray()[idx];
      auto [permutationDim, broadcastDim] = computeTransposeBroadcast(map);
      if (permutationDim.size() || broadcastDim.size()) {
        auto x = op.createLoopRanges(rewriter, op.getLoc());
        SmallVector<Value> dims;
        for (int i = 0; i < op.getNumLoops(); i++) {
          if (x[i].size.is<Value>()) {
            dims.push_back(x[i].size.get<Value>());
          }
        }
        Value broadcasted;
        if (op.hasPureTensorSemantics()) {
          broadcasted = rewriter.create<tensor::EmptyOp>(
              op.getLoc(),
              TypeRange{RankedTensorType::get(op.getStaticLoopRanges(),
                                              cast<ShapedType>(oper->get().getType()).getElementType())},
              dims);
        } else {
          broadcasted = rewriter.create<memref::AllocOp>(
              op.getLoc(),
              MemRefType::get(op.getStaticLoopRanges(), cast<ShapedType>(oper->get().getType()).getElementType()),
              dims);
        }
        Value currentValue = oper->get();
        if (permutationDim.size()) {
          SmallVector<int64_t> originShape(cast<ShapedType>(oper->get().getType()).getShape());
          applyPermutationToVector(originShape, permutationDim);
          SmallVector<Value> dynDims;
          for (auto [idx, s] : llvm::enumerate(cast<ShapedType>(oper->get().getType()).getShape())) {
            if (ShapedType::isDynamic(s)) {
              Value dim = op.hasPureTensorSemantics()
                              ? rewriter.create<tensor::DimOp>(op.getLoc(), currentValue, idx).getResult()
                              : rewriter.create<memref::DimOp>(op.getLoc(), currentValue, idx).getResult();
              dynDims.push_back(dim);
            }
          }
          Value emptyValue;
          if (op.hasPureTensorSemantics()) {
            emptyValue = rewriter.create<tensor::EmptyOp>(
                op.getLoc(),
                TypeRange{RankedTensorType::get(originShape, cast<ShapedType>(oper->get().getType()).getElementType())},
                dynDims);
          } else {
            emptyValue = rewriter.create<memref::AllocOp>(
                op.getLoc(), MemRefType::get(originShape, cast<ShapedType>(oper->get().getType()).getElementType()),
                dynDims);
          }
          auto transpose = rewriter.create<linalg::TransposeOp>(op.getLoc(), currentValue, emptyValue, permutationDim);
          if (op.hasPureTensorSemantics()) {
            currentValue = transpose.getResult()[0];
          } else {
            currentValue = emptyValue;
          }
        }
        auto broadcastOp = rewriter.create<linalg::BroadcastOp>(op.getLoc(), currentValue, broadcasted, broadcastDim);
        mapper.map(oper->get(), op.hasPureTensorSemantics() ? broadcastOp.getResult()[0] : broadcasted);
        newAttr[idx] = AffineMap::getMultiDimIdentityMap(op.getNumLoops(), op.getContext());
      }
    }
    if (mapper.getValueMap().size()) {
      auto newOp = rewriter.clone(*op, mapper);
      rewriter.replaceOp(op, newOp);
      auto newIndexingMaps = rewriter.getAffineMapArrayAttr(newAttr);
      newOp->setAttr("indexing_maps", newIndexingMaps);
    }
    return success(mapper.getValueMap().size() > 0);
  }
};

struct MatMulBroadcast : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    IRMapping mapper;
    if (op.hasUserDefinedMaps()) {
      auto ranges = dyn_cast<linalg::LinalgOp>(op.getOperation()).createLoopRanges(rewriter, loc);
      auto staticRange = op.getStaticLoopRanges();
      SmallVector<AffineMap> opIndexingMaps(op.getIndexingMapsArray());
      for (auto [idx, oper] : llvm::enumerate(op.getDpsInputOperands())) {
        AffineMap opIndexingMap = opIndexingMaps[idx];
        AffineMap defaultIndexingMap = op.getDefaultIndexingMaps(op->getContext())[idx];
        // auto x = getBroadcastAndTransposeFromMaps(opIndexingMap, defaultIndexingMap);
        auto shape = op.getShape(oper);
        if (opIndexingMap.getNumResults() < defaultIndexingMap.getNumResults()) {
          opIndexingMaps[idx] = defaultIndexingMap;
          Value emptyValue;
          SmallVector<Value> dynDims;
          SmallVector<int64_t> newShape;
          if (idx == 0) {
            newShape.assign({staticRange[0], staticRange[2]});
            if (staticRange[0] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[0].size));
            }
            if (staticRange[2] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[2].size));
            }
          } else {
            newShape.assign({staticRange[2], staticRange[1]});
            if (staticRange[2] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[2].size));
            }
            if (staticRange[1] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[1].size));
            }
          }
          if (op.hasPureTensorSemantics()) {
            emptyValue = rewriter.create<tensor::EmptyOp>(
                loc,
                TypeRange{RankedTensorType::get(newShape, cast<ShapedType>(oper->get().getType()).getElementType())},
                dynDims);
          } else {
            emptyValue = rewriter.create<memref::AllocOp>(
                loc, MemRefType::get(newShape, cast<ShapedType>(oper->get().getType()).getElementType()), dynDims);
          }
          SmallVector<int64_t> broadDim;
          broadDim.push_back(idx);
          auto broadcastOp = rewriter.create<linalg::BroadcastOp>(loc, oper->get(), emptyValue, broadDim);
          mapper.map(oper->get(), op.hasPureTensorSemantics() ? broadcastOp.getResult()[0] : emptyValue);
        }
      }
      if (mapper.getValueMap().size()) {
        auto newOp = rewriter.clone(*op, mapper);
        rewriter.replaceOp(op, newOp);
        newOp->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(opIndexingMaps));
      }
    }
    return success(mapper.getValueMap().size() > 0);
  }
};

template <typename OpTy>
struct BatchBroadcast : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    IRMapping mapper;
    if (op.hasUserDefinedMaps()) {
      auto ranges = dyn_cast<linalg::LinalgOp>(op.getOperation()).createLoopRanges(rewriter, loc);
      auto staticRange = op.getStaticLoopRanges();
      SmallVector<AffineMap> opIndexingMaps(op.getIndexingMapsArray());
      for (auto [idx, oper] : llvm::enumerate(op.getDpsInputOperands())) {
        AffineMap opIndexingMap = opIndexingMaps[idx];
        AffineMap defaultIndexingMap = op.getDefaultIndexingMaps(op->getContext())[idx];
        // auto x = getBroadcastAndTransposeFromMaps(opIndexingMap, defaultIndexingMap);
        auto shape = op.getShape(oper);
        if (opIndexingMap.getNumResults() < defaultIndexingMap.getNumResults()) {
          opIndexingMaps[idx] = defaultIndexingMap;
          Value emptyValue;
          SmallVector<Value> dynDims;
          SmallVector<int64_t> newShape;
          if (idx == 0) {
            newShape.assign({staticRange[0], staticRange[1], staticRange[3]});
            if (staticRange[0] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[0].size));
            }
            if (staticRange[1] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[1].size));
            }
            if (staticRange[3] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[3].size));
            }
          } else {
            newShape.assign({staticRange[0], staticRange[3], staticRange[2]});
            if (staticRange[0] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[0].size));
            }
            if (staticRange[3] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[3].size));
            }
            if (staticRange[2] == ShapedType::kDynamic) {
              dynDims.push_back(dyn_cast<Value>(ranges[2].size));
            }
          }
          if (op.hasPureTensorSemantics()) {
            emptyValue = rewriter.create<tensor::EmptyOp>(
                loc,
                TypeRange{RankedTensorType::get(newShape, cast<ShapedType>(oper->get().getType()).getElementType())},
                dynDims);
          } else {
            emptyValue = rewriter.create<memref::AllocOp>(
                loc, MemRefType::get(newShape, cast<ShapedType>(oper->get().getType()).getElementType()), dynDims);
          }
          SmallVector<int64_t> broadDim;
          if (opIndexingMap.getNumResults() == 1) {
            broadDim.push_back(0);
            broadDim.push_back(idx + 1);
          } else {
            if (idx == 0) {
              broadDim.push_back(opIndexingMap.getResult(0).isFunctionOfDim(0));
            } else {
              broadDim.push_back(opIndexingMap.getResult(0).isFunctionOfDim(0) * 2);
            }
          }
          auto broadcastOp = rewriter.create<linalg::BroadcastOp>(loc, oper->get(), emptyValue, broadDim);
          mapper.map(oper->get(), op.hasPureTensorSemantics() ? broadcastOp.getResult()[0] : emptyValue);
        }
      }
      if (mapper.getValueMap().size()) {
        auto newOp = rewriter.clone(*op, mapper);
        rewriter.replaceOp(op, newOp);
        newOp->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(opIndexingMaps));
      }
    }
    return success(mapper.getValueMap().size() > 0);
  }
};

void populateLinalgBroadcastPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgBroadcast>(patterns.getContext());
  // patterns.add<MatMulBroadcast>(patterns.getContext());
  // patterns.add<BatchBroadcast<linalg::BatchMatmulOp>>(patterns.getContext());
  // patterns.add<BatchBroadcast<linalg::BatchReduceMatmulOp>>(patterns.getContext());
}

struct LinalgAddBroadcastPass : public impl::LinalgAddBroadcastPassBase<LinalgAddBroadcastPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgBroadcastPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createLinalgAddBroadcastPass() {
  return std::make_unique<LinalgAddBroadcastPass>();
}

}  // namespace mlir
