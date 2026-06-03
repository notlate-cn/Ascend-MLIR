//===- SymbolConstraints.cpp - Ascend symbol constraints ------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace mlir::ascend::symbol {
namespace {

constexpr llvm::StringLiteral kSymNameKey = "sym_name";
constexpr llvm::StringLiteral kMembersKey = "members";
constexpr llvm::StringLiteral kValueKey = "value";
constexpr llvm::StringLiteral kDimKey = "dim";

RankedTensorType getRankedTensorType(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

bool isRankedTensor(Value value) {
  return static_cast<bool>(getRankedTensorType(value));
}

LogicalResult verifyDimInBounds(func::FuncOp func, DimRef ref) {
  RankedTensorType type = getRankedTensorType(ref.value);
  if (!type)
    return func.emitError() << kSymbolConstraintsAttr
                            << " member value must be a ranked tensor";
  if (ref.dim < 0 || ref.dim >= type.getRank())
    return func.emitError() << kSymbolConstraintsAttr
                            << " member dim " << ref.dim
                            << " is out of range for rank " << type.getRank();
  return success();
}

FailureOr<DimRef> parseDimRef(func::FuncOp func,
                              const ValueOrdinalMap &ordinals,
                              Attribute rawMember) {
  auto member = dyn_cast<DictionaryAttr>(rawMember);
  if (!member)
    return func.emitError()
           << kSymbolConstraintsAttr
           << " class member must be a dictionary attribute";

  auto valueAttr = dyn_cast_or_null<IntegerAttr>(member.get(kValueKey));
  if (!valueAttr)
    return func.emitError() << kSymbolConstraintsAttr
                            << " class member must include an integer value";
  auto dimAttr = dyn_cast_or_null<IntegerAttr>(member.get(kDimKey));
  if (!dimAttr)
    return func.emitError() << kSymbolConstraintsAttr
                            << " class member must include an integer dim";

  int64_t ordinal = valueAttr.getInt();
  FailureOr<Value> value = resolveValueOrdinal(func, ordinals, ordinal);
  if (failed(value))
    return failure();

  DimRef ref{*value, dimAttr.getInt()};
  if (failed(verifyDimInBounds(func, ref)))
    return failure();
  return ref;
}

} // namespace

const SymbolConstraintClass *SymbolConstraintTable::lookup(DimRef ref) const {
  for (const SymbolConstraintClass &klass : classes) {
    for (DimRef member : klass.members) {
      if (member == ref)
        return &klass;
    }
  }
  return nullptr;
}

bool SymbolConstraintTable::areEquivalent(DimRef lhs, DimRef rhs) const {
  if (lhs == rhs)
    return true;
  const SymbolConstraintClass *lhsClass = lookup(lhs);
  return lhsClass && lhsClass == lookup(rhs);
}

ValueOrdinalMap buildValueOrdinalMap(func::FuncOp func) {
  ValueOrdinalMap ordinals;
  int64_t nextOrdinal = 0;
  for (BlockArgument arg : func.getArguments())
    ordinals.try_emplace(arg, nextOrdinal++);

  func.walk([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      if (isRankedTensor(result))
        ordinals.try_emplace(result, nextOrdinal++);
    }
  });
  return ordinals;
}

FailureOr<Value> resolveValueOrdinal(func::FuncOp func,
                                     const ValueOrdinalMap &ordinals,
                                     int64_t ordinal) {
  if (ordinal < 0)
    return func.emitError() << kSymbolConstraintsAttr
                            << " value ordinal must be non-negative";

  for (const auto &entry : ordinals) {
    if (entry.second == ordinal)
      return entry.first;
  }
  return func.emitError() << kSymbolConstraintsAttr
                          << " references unknown value ordinal " << ordinal;
}

FailureOr<SerializedDimRef> serializeDimRef(func::FuncOp func,
                                            const ValueOrdinalMap &ordinals,
                                            DimRef ref) {
  auto it = ordinals.find(ref.value);
  if (it == ordinals.end())
    return func.emitError() << kSymbolConstraintsAttr
                            << " cannot serialize unregistered value";
  if (failed(verifyDimInBounds(func, ref)))
    return failure();
  return SerializedDimRef{it->second, ref.dim};
}

DictionaryAttr buildSerializedDimRefAttr(MLIRContext *context,
                                         SerializedDimRef ref) {
  Builder builder(context);
  return builder.getDictionaryAttr({
      builder.getNamedAttr(kValueKey,
                           builder.getI64IntegerAttr(ref.valueOrdinal)),
      builder.getNamedAttr(kDimKey, builder.getI64IntegerAttr(ref.dim)),
  });
}

DictionaryAttr buildSymbolClassAttr(MLIRContext *context, StringRef symName,
                                    ArrayRef<SerializedDimRef> members) {
  Builder builder(context);
  SmallVector<Attribute> memberAttrs;
  for (SerializedDimRef member : members)
    memberAttrs.push_back(buildSerializedDimRefAttr(context, member));
  return builder.getDictionaryAttr({
      builder.getNamedAttr(kSymNameKey, builder.getStringAttr(symName)),
      builder.getNamedAttr(kMembersKey, builder.getArrayAttr(memberAttrs)),
  });
}

ArrayAttr buildSymbolConstraintAttr(MLIRContext *context,
                                    ArrayRef<DictionaryAttr> classes) {
  SmallVector<Attribute> classAttrs;
  for (DictionaryAttr klass : classes)
    classAttrs.push_back(klass);
  return ArrayAttr::get(context, classAttrs);
}

FailureOr<SymbolConstraintTable> parseSymbolConstraintAttr(func::FuncOp func) {
  SymbolConstraintTable table;
  Attribute rawAttr = func->getAttr(kSymbolConstraintsAttr);
  if (!rawAttr)
    return table;

  auto constraintAttr = dyn_cast<ArrayAttr>(rawAttr);
  if (!constraintAttr)
    return func.emitError()
           << kSymbolConstraintsAttr << " must be an array attribute";

  ValueOrdinalMap ordinals = buildValueOrdinalMap(func);
  llvm::StringSet<> seenSymbols;
  llvm::DenseSet<DimRef> seenMembers;

  for (auto [classIndex, rawClass] : llvm::enumerate(constraintAttr)) {
    auto klassAttr = dyn_cast<DictionaryAttr>(rawClass);
    if (!klassAttr)
      return func.emitError()
             << kSymbolConstraintsAttr << " element " << classIndex
             << " must be a dictionary attribute";

    auto symName = dyn_cast_or_null<StringAttr>(klassAttr.get(kSymNameKey));
    if (!symName || symName.getValue().empty())
      return func.emitError()
             << kSymbolConstraintsAttr << " element " << classIndex
             << " must include a non-empty string sym_name";
    if (!seenSymbols.insert(symName.getValue()).second)
      return func.emitError()
             << kSymbolConstraintsAttr << " duplicate sym_name "
             << symName.getValue();

    auto membersAttr = dyn_cast_or_null<ArrayAttr>(klassAttr.get(kMembersKey));
    if (!membersAttr || membersAttr.empty())
      return func.emitError()
             << kSymbolConstraintsAttr << " class " << symName.getValue()
             << " must include non-empty members";

    SymbolConstraintClass klass;
    klass.symName = symName;
    for (Attribute rawMember : membersAttr) {
      FailureOr<DimRef> ref = parseDimRef(func, ordinals, rawMember);
      if (failed(ref))
        return failure();
      if (!seenMembers.insert(*ref).second)
        return func.emitError()
               << kSymbolConstraintsAttr << " duplicate member dim ref";
      klass.members.push_back(*ref);
    }
    table.classes.push_back(std::move(klass));
  }
  return table;
}

LogicalResult verifySymbolConstraintAttr(func::FuncOp func) {
  return success(succeeded(parseSymbolConstraintAttr(func)));
}

} // namespace mlir::ascend::symbol
