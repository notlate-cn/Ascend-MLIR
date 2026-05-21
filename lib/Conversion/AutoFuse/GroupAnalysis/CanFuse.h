#pragma once

#include "FusionGroup.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir::afir {

/// Determine the fusion kind between two groups.
FusionKind getFusionKind(const FusionGroup &g1, const FusionGroup &g2);

/// Check whether merging g1 and g2 would create a cycle in the group
/// dependency graph (transitive, multi-hop).  Exposed so the iterative fusion
/// driver can re-validate a pair against the *live* group state before applying
/// it — pairs are generated against a stale snapshot, so an earlier merge in
/// the same batch can turn a previously-safe pair into a cycle-creating one.
bool wouldCreateCycle(const FusionGroup &g1, const FusionGroup &g2,
                      llvm::ArrayRef<FusionGroup> allGroups);

/// Check if two VectorGroups can be fused.
bool canFuseVector(const FusionGroup &g1, const FusionGroup &g2,
                   FusionKind kind,
                   llvm::ArrayRef<FusionGroup> allGroups,
                   const CanFuseOptions &opts);

/// Check if a CubeGroup and a VectorGroup can be fused as epilogue.
bool canFuseCubeEpilogue(const FusionGroup &cube, const FusionGroup &vec,
                         llvm::ArrayRef<FusionGroup> allGroups);

/// Compute a fusion score (bytes of link tensors).
int64_t computeScore(const FusionGroup &g1, const FusionGroup &g2);

} // namespace mlir::afir
