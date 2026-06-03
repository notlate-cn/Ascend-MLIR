//===- SymbolConstraints.h - Ascend symbol constraints -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_COMMON_SYMBOLCONSTRAINTS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_COMMON_SYMBOLCONSTRAINTS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::ascend::symbol {

struct DimRef {
  Value value;
  int64_t dim = -1;

  friend bool operator==(const DimRef &lhs, const DimRef &rhs) {
    return lhs.value == rhs.value && lhs.dim == rhs.dim;
  }
};

struct SerializedDimRef {
  int64_t valueOrdinal = -1;
  int64_t dim = -1;
};

struct SymbolConstraintClass {
  StringAttr symName;
  SmallVector<DimRef, 4> members;
};

using ValueOrdinalMap = llvm::DenseMap<Value, int64_t>;

struct SymbolConstraintTable {
  SmallVector<SymbolConstraintClass, 4> classes;
  const SymbolConstraintClass *lookup(DimRef ref) const;
  bool areEquivalent(DimRef lhs, DimRef rhs) const;
};

ValueOrdinalMap buildValueOrdinalMap(func::FuncOp func);
FailureOr<Value> resolveValueOrdinal(func::FuncOp func,
                                     const ValueOrdinalMap &ordinals,
                                     int64_t ordinal);
FailureOr<SerializedDimRef> serializeDimRef(func::FuncOp func,
                                            const ValueOrdinalMap &ordinals,
                                            DimRef ref);
DictionaryAttr buildSerializedDimRefAttr(MLIRContext *context,
                                         SerializedDimRef ref);
DictionaryAttr buildSymbolClassAttr(MLIRContext *context, StringRef symName,
                                    ArrayRef<SerializedDimRef> members);
ArrayAttr buildSymbolConstraintAttr(MLIRContext *context,
                                    ArrayRef<DictionaryAttr> classes);
FailureOr<SymbolConstraintTable> parseSymbolConstraintAttr(func::FuncOp func);
LogicalResult verifySymbolConstraintAttr(func::FuncOp func);

} // namespace mlir::ascend::symbol

namespace llvm {
template <>
struct DenseMapInfo<mlir::ascend::symbol::DimRef> {
  static mlir::ascend::symbol::DimRef getEmptyKey() {
    return {DenseMapInfo<mlir::Value>::getEmptyKey(), -1};
  }
  static mlir::ascend::symbol::DimRef getTombstoneKey() {
    return {DenseMapInfo<mlir::Value>::getTombstoneKey(), -1};
  }
  static unsigned getHashValue(const mlir::ascend::symbol::DimRef &ref) {
    return hash_combine(ref.value, ref.dim);
  }
  static bool isEqual(const mlir::ascend::symbol::DimRef &lhs,
                      const mlir::ascend::symbol::DimRef &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // ASCEND_MLIR_CONVERSION_ASCEND_COMMON_SYMBOLCONSTRAINTS_H
