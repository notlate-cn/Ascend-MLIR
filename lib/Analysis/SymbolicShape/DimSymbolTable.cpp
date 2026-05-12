//===- DimSymbolTable.cpp - Per-function symbolic-dimension table ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Analysis/SymbolicShape/DimSymbolTable.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <limits>

using namespace mlir;
using namespace mlir::afir::symshape;

SymId DimSymbolTable::getOrCreateForArgDim(unsigned argIdx, unsigned dimIdx) {
  for (auto [i, e] : llvm::enumerate(entries))
    if (e.argIdx == argIdx && e.dimIdx == dimIdx)
      return static_cast<SymId>(i);
  SymId id = static_cast<SymId>(entries.size());
  entries.push_back({argIdx, dimIdx, id});
  return id;
}

SymId DimSymbolTable::find(SymId id) const {
  assert(id < entries.size() && "SymId out of range");
  while (entries[id].parent != id)
    id = entries[id].parent;
  return id;
}

void DimSymbolTable::alias(SymId a, SymId b) {
  SymId ra = find(a), rb = find(b);
  if (ra == rb)
    return;
  // Earlier-created root wins so serialized ids stay monotone under unions.
  if (ra < rb)
    entries[rb].parent = ra;
  else
    entries[ra].parent = rb;
}

std::pair<unsigned, unsigned> DimSymbolTable::sourceOf(SymId id) const {
  SymId r = find(id);
  return {entries[r].argIdx, entries[r].dimIdx};
}

llvm::SmallVector<unsigned, 8> DimSymbolTable::computeSerializedIds() const {
  llvm::SmallVector<unsigned, 8> serialized(entries.size(),
                                            std::numeric_limits<unsigned>::max());
  unsigned next = 0;
  for (unsigned i = 0, n = entries.size(); i < n; ++i)
    if (entries[i].parent == i)
      serialized[i] = next++;
  return serialized;
}

unsigned DimSymbolTable::numRoots() const {
  unsigned n = 0;
  for (unsigned i = 0, e = entries.size(); i < e; ++i)
    if (entries[i].parent == i)
      ++n;
  return n;
}

unsigned DimSymbolTable::serializedId(SymId id) const {
  auto serialized = computeSerializedIds();
  unsigned s = serialized[find(id)];
  assert(s != std::numeric_limits<unsigned>::max() && "root not serialized");
  return s;
}

SymExpr DimSymbolTable::toSerialized(SymExpr e) const {
  auto serialized = computeSerializedIds();
  return e.mapSymbols([&](SymId id) -> SymId {
    unsigned s = serialized[find(id)];
    assert(s != std::numeric_limits<unsigned>::max());
    return static_cast<SymId>(s);
  });
}

ArrayAttr DimSymbolTable::toAttr(MLIRContext *ctx) const {
  Builder b(ctx);
  auto serialized = computeSerializedIds();
  // Collect roots ordered by serialized id.
  llvm::SmallVector<unsigned, 8> rootBySerialized(numRoots(), 0);
  for (unsigned i = 0, n = entries.size(); i < n; ++i)
    if (entries[i].parent == i)
      rootBySerialized[serialized[i]] = i;

  llvm::SmallVector<Attribute, 8> dicts;
  for (unsigned s = 0; s < rootBySerialized.size(); ++s) {
    const Entry &e = entries[rootBySerialized[s]];
    dicts.push_back(b.getDictionaryAttr({
        b.getNamedAttr("id", b.getI64IntegerAttr(s)),
        b.getNamedAttr("arg", b.getI64IntegerAttr(e.argIdx)),
        b.getNamedAttr("dim", b.getI64IntegerAttr(e.dimIdx)),
    }));
  }
  return b.getArrayAttr(dicts);
}

std::optional<DimSymbolTable> DimSymbolTable::fromAttr(ArrayAttr attr) {
  if (!attr)
    return std::nullopt;
  DimSymbolTable t;
  for (auto [i, a] : llvm::enumerate(attr)) {
    auto dict = dyn_cast<DictionaryAttr>(a);
    if (!dict)
      return std::nullopt;
    auto idAttr = dict.getAs<IntegerAttr>("id");
    auto argAttr = dict.getAs<IntegerAttr>("arg");
    auto dimAttr = dict.getAs<IntegerAttr>("dim");
    if (!idAttr || !argAttr || !dimAttr)
      return std::nullopt;
    if (idAttr.getInt() != static_cast<int64_t>(i))
      return std::nullopt; // ids must be dense and in order
    if (argAttr.getInt() < 0 || dimAttr.getInt() < 0)
      return std::nullopt;
    t.entries.push_back({static_cast<unsigned>(argAttr.getInt()),
                         static_cast<unsigned>(dimAttr.getInt()),
                         static_cast<SymId>(i)});
  }
  return t;
}
