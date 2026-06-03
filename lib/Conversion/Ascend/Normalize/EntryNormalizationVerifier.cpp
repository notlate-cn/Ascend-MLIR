//===- EntryNormalizationVerifier.cpp - Normalize verifier ---------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "EntryNormalizationVerifier.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "SymbolEquivalenceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;

namespace mlir::ascend::normalize {
namespace {

constexpr llvm::StringLiteral kMembersKey = "members";
constexpr llvm::StringLiteral kValueKey = "value";
constexpr llvm::StringLiteral kDimKey = "dim";

bool isStaticRankedTensorDim(symbol::DimRef ref) {
  auto type = dyn_cast<RankedTensorType>(ref.value.getType());
  return type && !type.isDynamicDim(ref.dim);
}

class ExpectedEquivalenceClosure {
public:
  void addEquality(symbol::DimRef lhs, symbol::DimRef rhs) {
    if (lhs == rhs || isStaticRankedTensorDim(lhs) ||
        isStaticRankedTensorDim(rhs))
      return;

    unsigned lhsId = getOrCreate(lhs);
    unsigned rhsId = getOrCreate(rhs);
    unite(lhsId, rhsId);
  }

  bool areEquivalent(symbol::DimRef lhs, symbol::DimRef rhs) {
    if (lhs == rhs)
      return true;
    auto lhsId = ids.find(lhs);
    auto rhsId = ids.find(rhs);
    return lhsId != ids.end() && rhsId != ids.end() &&
           find(lhsId->second) == find(rhsId->second);
  }

private:
  unsigned getOrCreate(symbol::DimRef ref) {
    auto [it, inserted] = ids.try_emplace(ref, parents.size());
    if (inserted) {
      parents.push_back(it->second);
      ranks.push_back(0);
    }
    return it->second;
  }

  unsigned find(unsigned id) {
    if (parents[id] == id)
      return id;
    parents[id] = find(parents[id]);
    return parents[id];
  }

  void unite(unsigned lhsId, unsigned rhsId) {
    unsigned lhsRoot = find(lhsId);
    unsigned rhsRoot = find(rhsId);
    if (lhsRoot == rhsRoot)
      return;
    if (ranks[lhsRoot] < ranks[rhsRoot])
      std::swap(lhsRoot, rhsRoot);
    parents[rhsRoot] = lhsRoot;
    if (ranks[lhsRoot] == ranks[rhsRoot])
      ++ranks[lhsRoot];
  }

  llvm::DenseMap<symbol::DimRef, unsigned> ids;
  SmallVector<unsigned, 16> parents;
  SmallVector<unsigned, 16> ranks;
};

FailureOr<int64_t> parseGeneratedI64Field(func::FuncOp func,
                                          DictionaryAttr dict,
                                          StringRef fieldName) {
  auto intAttr = dyn_cast_or_null<IntegerAttr>(dict.get(fieldName));
  if (!intAttr)
    return func.emitError() << "generated symbol constraints field "
                            << fieldName << " must be i64";
  return intAttr.getValue().getSExtValue();
}

FailureOr<symbol::DimRef> parseGeneratedDimRef(
    func::FuncOp func, const symbol::ValueOrdinalMap &ordinals,
    Attribute rawMember) {
  auto member = dyn_cast<DictionaryAttr>(rawMember);
  if (!member)
    return func.emitError()
           << "generated symbol constraints member must be a dictionary";

  FailureOr<int64_t> ordinal =
      parseGeneratedI64Field(func, member, kValueKey);
  if (failed(ordinal))
    return failure();
  FailureOr<int64_t> dim = parseGeneratedI64Field(func, member, kDimKey);
  if (failed(dim))
    return failure();

  FailureOr<Value> value =
      symbol::resolveValueOrdinal(func, ordinals, *ordinal);
  if (failed(value))
    return failure();
  return symbol::DimRef{*value, *dim};
}

FailureOr<llvm::DenseSet<symbol::DimRef>>
collectExpectedSingletonMembers(func::FuncOp func, ArrayAttr expectedAttr) {
  symbol::ValueOrdinalMap ordinals = symbol::buildValueOrdinalMap(func);
  llvm::DenseSet<symbol::DimRef> singletons;

  for (Attribute rawClass : expectedAttr) {
    auto klass = dyn_cast<DictionaryAttr>(rawClass);
    if (!klass)
      return func.emitError()
             << "generated symbol constraints class must be a dictionary";

    auto members = dyn_cast_or_null<ArrayAttr>(klass.get(kMembersKey));
    if (!members)
      return func.emitError()
             << "generated symbol constraints class members must be an array";
    if (members.size() != 1)
      continue;

    FailureOr<symbol::DimRef> ref =
        parseGeneratedDimRef(func, ordinals, *members.begin());
    if (failed(ref))
      return failure();
    singletons.insert(*ref);
  }
  return singletons;
}

void buildExpectedClosure(const SymbolEquivalenceResult &expected,
                          ExpectedEquivalenceClosure &closure) {
  for (const SymbolEqualityProof &proof : expected.proofs)
    closure.addEquality(proof.lhs, proof.rhs);
}

LogicalResult verifyExpectedProofCoverage(
    func::FuncOp func, const symbol::SymbolConstraintTable &table,
    const SymbolEquivalenceResult &expected) {
  LogicalResult result = success();
  for (const SymbolEqualityProof &proof : expected.proofs) {
    if (isStaticRankedTensorDim(proof.lhs) ||
        isStaticRankedTensorDim(proof.rhs))
      continue;
    if (table.areEquivalent(proof.lhs, proof.rhs))
      continue;

    func.emitError() << "symbol constraints missing " << proof.rule
                     << " equality proof";
    result = failure();
  }
  return result;
}

LogicalResult verifyNoExtraSymbolConstraints(
    func::FuncOp func, const symbol::SymbolConstraintTable &table,
    ExpectedEquivalenceClosure &closure,
    const llvm::DenseSet<symbol::DimRef> &expectedSingletonMembers) {
  LogicalResult result = success();
  for (const symbol::SymbolConstraintClass &klass : table.classes) {
    SmallVector<symbol::DimRef, 4> dynamicMembers;
    for (symbol::DimRef member : klass.members) {
      if (isStaticRankedTensorDim(member)) {
        func.emitError()
            << "symbol constraints class " << klass.symName.getValue()
            << " contains static dim member";
        result = failure();
        continue;
      }
      dynamicMembers.push_back(member);
    }

    if (dynamicMembers.size() == 1 &&
        !expectedSingletonMembers.contains(dynamicMembers.front())) {
      func.emitError() << "symbol constraints class "
                       << klass.symName.getValue()
                       << " contains unjustified singleton member";
      result = failure();
    }

    bool reportedOverMerge = false;
    for (auto [index, lhs] : llvm::enumerate(dynamicMembers)) {
      for (symbol::DimRef rhs : ArrayRef(dynamicMembers).drop_front(index + 1)) {
        if (closure.areEquivalent(lhs, rhs))
          continue;
        if (!reportedOverMerge) {
          func.emitError() << "symbol constraints class "
                           << klass.symName.getValue()
                           << " over-merges unrelated dynamic dims";
          reportedOverMerge = true;
        }
        result = failure();
      }
    }
  }
  return result;
}

LogicalResult verifyFuncSymbolConstraints(func::FuncOp func) {
  if (!func->getAttr(kSymbolConstraintsAttr))
    return func.emitError() << kSymbolConstraintsAttr << " missing";

  FailureOr<symbol::SymbolConstraintTable> table =
      symbol::parseSymbolConstraintAttr(func);
  if (failed(table))
    return failure();

  FailureOr<SymbolEquivalenceResult> expected =
      analyzeSymbolEquivalence(func);
  if (failed(expected))
    return func.emitError("failed to recompute symbol equivalence proofs");

  ExpectedEquivalenceClosure expectedClosure;
  buildExpectedClosure(*expected, expectedClosure);

  FailureOr<llvm::DenseSet<symbol::DimRef>> expectedSingletonMembers =
      collectExpectedSingletonMembers(func, expected->attr);
  if (failed(expectedSingletonMembers))
    return failure();

  LogicalResult result = success();
  if (failed(verifyExpectedProofCoverage(func, *table, *expected)))
    result = failure();
  if (failed(verifyNoExtraSymbolConstraints(
          func, *table, expectedClosure, *expectedSingletonMembers)))
    result = failure();
  return result;
}

} // namespace

LogicalResult verifyEntryNormalization(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](func::FuncOp func) {
    if (failed(verifyFuncSymbolConstraints(func)))
      result = failure();
  });
  return result;
}

} // namespace mlir::ascend::normalize
