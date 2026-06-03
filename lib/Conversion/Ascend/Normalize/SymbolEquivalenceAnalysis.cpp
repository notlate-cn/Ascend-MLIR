//===- SymbolEquivalenceAnalysis.cpp - Normalize symbol facts ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "SymbolEquivalenceAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include <algorithm>
#include <optional>
#include <tuple>

using namespace mlir;

namespace mlir::ascend::normalize {
namespace {

RankedTensorType getRankedTensorType(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

bool isRankedTensor(Value value) {
  return static_cast<bool>(getRankedTensorType(value));
}

bool isStaticDim(symbol::DimRef ref) {
  RankedTensorType type = getRankedTensorType(ref.value);
  return type && !type.isDynamicDim(ref.dim);
}

LogicalResult verifyDimRef(func::FuncOp func, symbol::DimRef ref) {
  RankedTensorType type = getRankedTensorType(ref.value);
  if (!type)
    return func.emitError()
           << "Normalize symbol equivalence expected ranked tensor value";
  if (ref.dim < 0 || ref.dim >= type.getRank())
    return func.emitError()
           << "Normalize symbol equivalence dim " << ref.dim
           << " is out of range for rank " << type.getRank();
  return success();
}

std::optional<int64_t> getConstantIndex(OpFoldResult value) {
  return getConstantIntValue(value);
}

std::optional<int64_t> getConstantIndex(Value value) {
  return getConstantIntValue(OpFoldResult(value));
}

class SymbolUnionFind {
public:
  LogicalResult addEquality(func::FuncOp func, symbol::DimRef lhs,
                            symbol::DimRef rhs, StringRef rule) {
    if (failed(verifyDimRef(func, lhs)) || failed(verifyDimRef(func, rhs)))
      return failure();

    unsigned lhsId = getOrCreate(lhs);
    unsigned rhsId = getOrCreate(rhs);
    proofs.push_back(SymbolEqualityProof{lhs, rhs, rule});
    unite(lhsId, rhsId);
    return success();
  }

  unsigned find(unsigned id) {
    if (parents[id] == id)
      return id;
    unsigned parent = parents[id];
    unsigned root = find(parent);
    weights[id] += weights[parent];
    parents[id] = root;
    return root;
  }

  std::optional<unsigned> lookup(symbol::DimRef ref) const {
    auto it = ids.find(ref);
    if (it == ids.end())
      return std::nullopt;
    return it->second;
  }

  ArrayRef<symbol::DimRef> getNodes() const { return nodes; }

  ArrayRef<SymbolEqualityProof> getProofs() const { return proofs; }

private:
  unsigned getOrCreate(symbol::DimRef ref) {
    auto [it, inserted] = ids.try_emplace(ref, nodes.size());
    if (!inserted)
      return it->second;

    nodes.push_back(ref);
    parents.push_back(it->second);
    ranks.push_back(0);
    weights.push_back(0);
    return it->second;
  }

  void unite(unsigned lhsId, unsigned rhsId) {
    unsigned lhsRoot = find(lhsId);
    unsigned rhsRoot = find(rhsId);
    if (lhsRoot == rhsRoot)
      return;

    if (ranks[lhsRoot] < ranks[rhsRoot])
      std::swap(lhsRoot, rhsRoot);
    parents[rhsRoot] = lhsRoot;
    weights[rhsRoot] = 0;
    if (ranks[lhsRoot] == ranks[rhsRoot])
      ++ranks[lhsRoot];
  }

  llvm::DenseMap<symbol::DimRef, unsigned> ids;
  SmallVector<symbol::DimRef, 16> nodes;
  SmallVector<unsigned, 16> parents;
  SmallVector<unsigned, 16> ranks;
  SmallVector<int64_t, 16> weights;
  SmallVector<SymbolEqualityProof, 16> proofs;
};

struct AnalysisState {
  explicit AnalysisState(func::FuncOp func) : func(func) {}

  func::FuncOp func;
  SymbolUnionFind uf;
  llvm::DenseMap<Value, symbol::DimRef> tensorDimResults;

  LogicalResult addEquality(symbol::DimRef lhs, symbol::DimRef rhs,
                            StringRef rule) {
    return uf.addEquality(func, lhs, rhs, rule);
  }

  LogicalResult addDimEquality(Value lhsValue, int64_t lhsDim, Value rhsValue,
                               int64_t rhsDim, StringRef rule) {
    if (!isRankedTensor(lhsValue) || !isRankedTensor(rhsValue))
      return success();
    return addEquality(symbol::DimRef{lhsValue, lhsDim},
                       symbol::DimRef{rhsValue, rhsDim}, rule);
  }

  LogicalResult addSameRankEquality(Value lhsValue, Value rhsValue,
                                    StringRef rule) {
    RankedTensorType lhsType = getRankedTensorType(lhsValue);
    RankedTensorType rhsType = getRankedTensorType(rhsValue);
    if (!lhsType || !rhsType)
      return success();
    if (lhsType.getRank() != rhsType.getRank())
      return func.emitError()
             << "Normalize symbol equivalence expected equal tensor ranks";
    for (int64_t dim = 0, rank = lhsType.getRank(); dim < rank; ++dim)
      if (failed(addDimEquality(lhsValue, dim, rhsValue, dim, rule)))
        return failure();
    return success();
  }
};

LogicalResult collectTensorDimResults(AnalysisState &state) {
  WalkResult walkResult = state.func.walk([&](tensor::DimOp dimOp) {
    std::optional<int64_t> dim = getConstantIndex(dimOp.getIndex());
    if (!dim)
      return WalkResult::advance();

    Value source = dimOp.getSource();
    if (!isRankedTensor(source))
      return WalkResult::advance();
    symbol::DimRef ref{source, *dim};
    if (failed(verifyDimRef(state.func, ref)))
      return WalkResult::interrupt();
    state.tensorDimResults.try_emplace(dimOp.getResult(), ref);
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

LogicalResult addDpsResultInitEqualities(AnalysisState &state,
                                         linalg::LinalgOp linalgOp) {
  for (OpResult result : linalgOp->getResults()) {
    OpOperand *init = linalgOp.getDpsInitOperand(result.getResultNumber());
    if (!init)
      continue;
    if (failed(state.addSameRankEquality(init->get(), result,
                                         "dps_result_init")))
      return failure();
  }
  return success();
}

LogicalResult analyzeProjectedDimEqualities(AnalysisState &state,
                                            linalg::LinalgOp linalgOp,
                                            StringRef rule) {
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  SmallVector<Value> operands;
  llvm::append_range(operands, linalgOp.getDpsInputs());
  llvm::append_range(operands, linalgOp.getDpsInits());

  if (maps.size() != operands.size())
    return linalgOp->emitError()
           << "Normalize symbol equivalence expected one indexing map per "
              "linalg operand";

  llvm::DenseMap<unsigned, SmallVector<symbol::DimRef, 4>> refsByIterator;
  for (auto [operandIndex, operand] : llvm::enumerate(operands)) {
    RankedTensorType type = getRankedTensorType(operand);
    if (!type)
      continue;

    AffineMap map = maps[operandIndex];
    if (map.getNumResults() != static_cast<unsigned>(type.getRank()))
      return linalgOp->emitError()
             << "Normalize symbol equivalence expected indexing map rank to "
                "match operand rank";

    llvm::DenseMap<unsigned, int64_t> dimByIterator;
    llvm::DenseSet<unsigned> duplicateIterators;
    for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        continue;
      unsigned iterator = dimExpr.getPosition();
      auto insertion =
          dimByIterator.try_emplace(iterator, static_cast<int64_t>(dim));
      if (!insertion.second)
        duplicateIterators.insert(iterator);
    }

    for (auto &entry : dimByIterator) {
      if (duplicateIterators.contains(entry.first))
        continue;
      refsByIterator[entry.first].push_back(
          symbol::DimRef{operand, entry.second});
    }
  }

  for (auto &entry : refsByIterator) {
    SmallVectorImpl<symbol::DimRef> &refs = entry.second;
    for (unsigned i = 1, e = refs.size(); i < e; ++i)
      if (failed(state.addEquality(refs.front(), refs[i], rule)))
        return failure();
  }
  return success();
}

LogicalResult analyzeProducerConsumerEdges(AnalysisState &state,
                                           Operation *op) {
  for (Value operand : op->getOperands()) {
    auto result = dyn_cast<OpResult>(operand);
    if (!result || !isRankedTensor(result))
      continue;

    RankedTensorType type = getRankedTensorType(result);
    for (int64_t dim = 0, rank = type.getRank(); dim < rank; ++dim)
      if (failed(state.addEquality(symbol::DimRef{result, dim},
                                   symbol::DimRef{result, dim}, "r3_ssa")))
        return failure();
  }
  return success();
}

bool isTensorDimOf(Value value, Value source, int64_t dim) {
  auto dimOp = value.getDefiningOp<tensor::DimOp>();
  if (!dimOp || dimOp.getSource() != source)
    return false;
  std::optional<int64_t> dimIndex = getConstantIndex(dimOp.getIndex());
  return dimIndex && *dimIndex == dim;
}

bool isFullStaticSize(RankedTensorType sourceType, int64_t sourceDim,
                      OpFoldResult size) {
  if (sourceType.isDynamicDim(sourceDim))
    return false;
  int64_t staticSourceDim = sourceType.getDimSize(sourceDim);
  if (staticSourceDim == 1)
    return false;
  std::optional<int64_t> staticSize = getConstantIndex(size);
  return staticSize && *staticSize == staticSourceDim;
}

LogicalResult analyzeExtractSlice(AnalysisState &state,
                                  tensor::ExtractSliceOp slice) {
  Value source = slice.getSource();
  Value result = slice.getResult();
  RankedTensorType sourceType = getRankedTensorType(source);
  RankedTensorType resultType = getRankedTensorType(result);
  if (!sourceType || !resultType)
    return success();

  SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
  SmallVector<OpFoldResult> strides = slice.getMixedStrides();
  llvm::SmallBitVector droppedDims = slice.getDroppedDims();

  int64_t resultDim = 0;
  for (int64_t sourceDim = 0, sourceRank = sourceType.getRank();
       sourceDim < sourceRank; ++sourceDim) {
    if (droppedDims.test(sourceDim))
      continue;
    if (resultDim >= resultType.getRank())
      return slice.emitError()
             << "Normalize symbol equivalence invalid extract_slice rank map";

    bool offsetZero = getConstantIndex(offsets[sourceDim]) == int64_t{0};
    bool strideOne = getConstantIndex(strides[sourceDim]) == int64_t{1};
    if (offsetZero && strideOne) {
      OpFoldResult size = sizes[sourceDim];
      auto sizeValue = size.dyn_cast<Value>();
      bool fullSize =
          isFullStaticSize(sourceType, sourceDim, size) ||
          (sizeValue && isTensorDimOf(sizeValue, source, sourceDim));
      if (fullSize &&
          failed(state.addDimEquality(source, sourceDim, result, resultDim,
                                      "r4_extract_slice")))
        return failure();

      if (sizeValue) {
        auto it = state.tensorDimResults.find(sizeValue);
        if (it != state.tensorDimResults.end() &&
            failed(state.addEquality(
                it->second, symbol::DimRef{result, resultDim},
                "r5_extract_slice_size")))
          return failure();
      }
    }

    ++resultDim;
  }
  return success();
}

LogicalResult analyzeEmpty(AnalysisState &state, tensor::EmptyOp empty) {
  Value result = empty.getResult();
  RankedTensorType resultType = getRankedTensorType(result);
  if (!resultType)
    return success();

  SmallVector<OpFoldResult> sizes = empty.getMixedSizes();
  for (int64_t dim = 0, rank = resultType.getRank(); dim < rank; ++dim) {
    auto sizeValue = sizes[dim].dyn_cast<Value>();
    if (!sizeValue)
      continue;
    auto it = state.tensorDimResults.find(sizeValue);
    if (it == state.tensorDimResults.end())
      continue;
    if (failed(state.addEquality(it->second, symbol::DimRef{result, dim},
                                 "r5_tensor_empty")))
      return failure();
  }
  return success();
}

LogicalResult analyzeBroadcast(AnalysisState &state,
                               linalg::BroadcastOp broadcast) {
  auto inputs = broadcast.getDpsInputs();
  auto outputs = broadcast.getDpsInits();
  if (inputs.empty() || outputs.empty())
    return success();

  Value input = inputs[0];
  Value output = outputs[0];
  RankedTensorType inputType = getRankedTensorType(input);
  RankedTensorType outputType = getRankedTensorType(output);
  if (!inputType || !outputType)
    return success();

  auto dimensions = broadcast->getAttrOfType<DenseI64ArrayAttr>("dimensions");
  if (!dimensions)
    return broadcast.emitError()
           << "Normalize symbol equivalence expected linalg.broadcast "
              "dimensions attr";

  llvm::DenseSet<int64_t> broadcastDims;
  for (int64_t dim : dimensions.asArrayRef())
    broadcastDims.insert(dim);

  int64_t inputDim = 0;
  for (int64_t outputDim = 0, outputRank = outputType.getRank();
       outputDim < outputRank; ++outputDim) {
    if (broadcastDims.contains(outputDim))
      continue;
    if (inputDim >= inputType.getRank())
      return broadcast.emitError()
             << "Normalize symbol equivalence invalid broadcast rank map";
    if (failed(state.addDimEquality(input, inputDim, output, outputDim,
                                    "r6_linalg_broadcast")))
      return failure();
    ++inputDim;
  }
  if (inputDim != inputType.getRank())
    return broadcast.emitError()
           << "Normalize symbol equivalence invalid broadcast dimensions";
  return success();
}

LogicalResult analyzeOperation(AnalysisState &state, Operation *op) {
  if (failed(analyzeProducerConsumerEdges(state, op)))
    return failure();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    if (failed(addDpsResultInitEqualities(state, linalgOp)))
      return failure();

  if (auto generic = dyn_cast<linalg::GenericOp>(op))
    return analyzeProjectedDimEqualities(state, generic, "r1_linalg_generic");
  if (isa<linalg::MatmulOp>(op))
    return analyzeProjectedDimEqualities(state, cast<linalg::LinalgOp>(op),
                                         "r2_linalg_matmul");
  if (isa<linalg::BatchMatmulOp>(op))
    return analyzeProjectedDimEqualities(state, cast<linalg::LinalgOp>(op),
                                         "r2_linalg_batch_matmul");
  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op))
    return analyzeExtractSlice(state, slice);
  if (auto empty = dyn_cast<tensor::EmptyOp>(op))
    return analyzeEmpty(state, empty);
  if (auto broadcast = dyn_cast<linalg::BroadcastOp>(op))
    return analyzeBroadcast(state, broadcast);

  return success();
}

struct MemberEntry {
  symbol::DimRef ref;
  symbol::SerializedDimRef serialized;
};

bool memberLess(const MemberEntry &lhs, const MemberEntry &rhs) {
  return std::tie(lhs.serialized.valueOrdinal, lhs.serialized.dim) <
         std::tie(rhs.serialized.valueOrdinal, rhs.serialized.dim);
}

bool proofSurvivesStaticFiltering(SymbolUnionFind &uf, unsigned root,
                                  ArrayRef<SymbolEqualityProof> proofs) {
  for (const SymbolEqualityProof &proof : proofs) {
    if (proof.lhs == proof.rhs || isStaticDim(proof.lhs) ||
        isStaticDim(proof.rhs))
      continue;
    std::optional<unsigned> lhsId = uf.lookup(proof.lhs);
    std::optional<unsigned> rhsId = uf.lookup(proof.rhs);
    if (lhsId && rhsId && uf.find(*lhsId) == root && uf.find(*rhsId) == root)
      return true;
  }
  return false;
}

std::optional<std::string> getArgumentSymbolName(func::FuncOp func,
                                                 ArrayRef<MemberEntry> members) {
  for (const MemberEntry &member : members) {
    auto arg = dyn_cast<BlockArgument>(member.ref.value);
    if (!arg || arg.getOwner()->getParentOp() != func.getOperation())
      continue;
    return (llvm::Twine("arg") + llvm::Twine(arg.getArgNumber()) +
            llvm::Twine("_dim") + llvm::Twine(member.ref.dim))
        .str();
  }
  return std::nullopt;
}

FailureOr<ArrayAttr> serializeConstraints(AnalysisState &state) {
  symbol::ValueOrdinalMap ordinals = symbol::buildValueOrdinalMap(state.func);
  llvm::DenseMap<unsigned, SmallVector<MemberEntry, 4>> membersByRoot;

  for (auto [id, ref] : llvm::enumerate(state.uf.getNodes())) {
    if (isStaticDim(ref))
      continue;
    FailureOr<symbol::SerializedDimRef> serialized =
        symbol::serializeDimRef(state.func, ordinals, ref);
    if (failed(serialized))
      return failure();
    membersByRoot[state.uf.find(id)].push_back(MemberEntry{ref, *serialized});
  }

  SmallVector<SmallVector<MemberEntry, 4>, 8> classes;
  for (auto &entry : membersByRoot) {
    SmallVector<MemberEntry, 4> members = std::move(entry.second);
    llvm::sort(members, memberLess);
    if (members.size() <= 1 &&
        !proofSurvivesStaticFiltering(state.uf, entry.first,
                                      state.uf.getProofs()))
      continue;
    classes.push_back(std::move(members));
  }

  llvm::sort(classes, [](ArrayRef<MemberEntry> lhs,
                         ArrayRef<MemberEntry> rhs) {
    assert(!lhs.empty() && !rhs.empty());
    return memberLess(lhs.front(), rhs.front());
  });

  MLIRContext *context = state.func.getContext();
  SmallVector<DictionaryAttr, 8> classAttrs;
  for (auto [classIndex, members] : llvm::enumerate(classes)) {
    SmallVector<symbol::SerializedDimRef, 4> serializedMembers;
    for (const MemberEntry &member : members)
      serializedMembers.push_back(member.serialized);

    std::string symName =
        getArgumentSymbolName(state.func, members)
            .value_or((llvm::Twine("sym_") + llvm::Twine(classIndex)).str());
    classAttrs.push_back(
        symbol::buildSymbolClassAttr(context, symName, serializedMembers));
  }
  return symbol::buildSymbolConstraintAttr(context, classAttrs);
}

} // namespace

FailureOr<SymbolEquivalenceResult>
analyzeSymbolEquivalence(func::FuncOp func) {
  AnalysisState state(func);
  if (failed(collectTensorDimResults(state)))
    return failure();

  WalkResult walkResult = func.walk([&](Operation *op) {
    if (failed(analyzeOperation(state, op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  FailureOr<ArrayAttr> attr = serializeConstraints(state);
  if (failed(attr))
    return failure();

  SymbolEquivalenceResult result;
  result.attr = *attr;
  llvm::append_range(result.proofs, state.uf.getProofs());
  return result;
}

} // namespace mlir::ascend::normalize
