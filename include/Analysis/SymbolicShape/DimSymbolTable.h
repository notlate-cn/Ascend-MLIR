//===- DimSymbolTable.h - Per-function symbolic-dimension table -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Owns the SymId namespace for one func::FuncOp.  Each symbol is "the d-th
// dimension of the a-th block argument" of @kernel.  Carries a union-find so
// dimensions discovered to be equal (e.g. two operands pinned to the same
// linalg iteration dim) collapse to one root symbol.  Serializes to / from the
// `afir.dim_symbols` ArrayAttr (roots only, re-indexed 0..numRoots-1).
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ANALYSIS_SYMBOLICSHAPE_SYMBOLTABLE_H
#define AFIR_ANALYSIS_SYMBOLICSHAPE_SYMBOLTABLE_H

#include "Analysis/SymbolicShape/SymExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <utility>

namespace mlir::afir::symshape {

class DimSymbolTable {
public:
  DimSymbolTable() = default;

  /// Returns the SymId for arg `argIdx` dim `dimIdx`, creating it if new.  The
  /// returned id is *not* canonicalized -- call find() to get the root.
  SymId getOrCreateForArgDim(unsigned argIdx, unsigned dimIdx);

  /// Unions the classes of `a` and `b`.  The earlier-created root wins, so
  /// already-serialized references stay valid as long as you only ever union.
  void alias(SymId a, SymId b);

  /// Canonical representative of `id`'s class (path-compressing).
  SymId find(SymId id) const;

  /// (argIdx, dimIdx) of `id`'s canonical root.
  std::pair<unsigned, unsigned> sourceOf(SymId id) const;

  /// Number of distinct root symbols.
  unsigned numRoots() const;

  /// Maps a SymId to its dense serialized id in [0, numRoots()).  This is the
  /// `s<k>` index that appears in `afir.symbolic_shapes` strings.
  unsigned serializedId(SymId id) const;

  /// Rewrites a SymExpr so every Sym leaf uses serialized ids (call before
  /// printing into an attribute).
  SymExpr toSerialized(SymExpr e) const;

  /// Builds the `afir.dim_symbols` ArrayAttr: one DictionaryAttr per root,
  /// `{id = <serialized> : i64, arg = <argIdx> : i64, dim = <dimIdx> : i64}`,
  /// ordered by serialized id.
  ArrayAttr toAttr(MLIRContext *ctx) const;

  /// Parses an `afir.dim_symbols` ArrayAttr (as produced by toAttr) back into a
  /// table whose SymIds equal the serialized ids.  Returns std::nullopt on a
  /// malformed attribute.
  static std::optional<DimSymbolTable> fromAttr(ArrayAttr attr);

private:
  struct Entry {
    unsigned argIdx;
    unsigned dimIdx;
    SymId parent; // == own index when root
  };
  llvm::SmallVector<Entry, 8> entries;

  /// root SymId -> dense serialized id, in creation order of roots.
  llvm::SmallVector<unsigned, 8> computeSerializedIds() const;
};

} // namespace mlir::afir::symshape

#endif // AFIR_ANALYSIS_SYMBOLICSHAPE_SYMBOLTABLE_H
