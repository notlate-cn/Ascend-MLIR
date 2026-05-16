//===- HandwrittenContractRegistry.h - Handwritten pattern contracts --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENCONTRACTREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENCONTRACTREGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::afir::ascend::kernelize {

/// Describes all compile-time behavior of one handwritten kernel pattern.
/// Register via registerHandwrittenContract(); consume via lookupHandwrittenContract().
struct HandwrittenContract {
  /// Minimum number of Cube ops required in the pattern graph.
  unsigned minCubeCount = 1;
  /// Maximum number of Cube ops allowed in the pattern graph.
  unsigned maxCubeCount = 1;
  /// Pattern must contain at least one Reduction op.
  bool requiresReduction = false;
  /// Pattern must contain at least one Vector+Injective op.
  bool requiresVectorInjective = false;

  /// AxisCoalescer: when true, only the dominant primary op contributes axes
  /// (the other ops' indexing maps are not merged into the coalesced axis set).
  bool useAxisCarrierOnly = false;

  /// Primary op selection: when non-empty, prefer the op whose Kernelize role
  /// matches this string (e.g. "Cube"). Falls back to priority-based selection.
  std::string primarySelectionRole;

  /// ScheduleProblemBuilder: extra structure constraint tags appended to
  /// ScheduleProblem::structureConstraints for this pattern.
  SmallVector<std::string, 2> structureConstraints;

  /// TemplateRegistry: the schedule template registered for this pattern.
  /// Fields: name, layout, tags, minRank, maxRank, priority.
  struct TemplateSpec {
    std::string name;
    std::string layout;
    SmallVector<std::string, 4> tags;
    unsigned minRank = 0;
    unsigned maxRank = 8;
    int priority = 0;
  };
  TemplateSpec scheduleTemplate;
};

/// Register a handwritten contract for the given kind string.
/// Must be called before any use of lookupHandwrittenContract.
/// Safe to call multiple times with the same kind (second call is a no-op).
void registerHandwrittenContract(llvm::StringRef kind,
                                 HandwrittenContract contract);

/// Register all built-in contracts (currently: attention_sdpa).
/// Called automatically on first lookup; exposed for test initialization.
void registerBuiltinHandwrittenContracts();

/// Returns nullptr if kind is empty or not registered.
const HandwrittenContract *lookupHandwrittenContract(llvm::StringRef kind);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENCONTRACTREGISTRY_H
