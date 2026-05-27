//===- OpRoleClassifier.h - Classify a linalg/arith op to a role -*- C++ -*-=//
//
// Returns a stable, human-readable role name for a single source op.
// Used by NetworkJsonEmitter to populate provenance's `op_role` and
// `fused_ops_summary` fields (docs/auto-fuse/debug.md §7.5).
//
// Coverage (in priority order):
//   1. linalg::MatmulOp / BatchMatmulOp                  → "matmul"
//   2. linalg::TransposeOp                               → "transpose"
//   3. linalg::CopyOp / FillOp / BroadcastOp             → "copy" / "fill" / "broadcast"
//   4. linalg::GenericOp, reduction iter                 → "reduce_<combiner>"
//                                                          combiner ∈ {sum, max, min, prod, ...}
//   5. linalg::GenericOp, single arith/math body op      → that op's short name ("add", "exp", "select", ...)
//   6. linalg::GenericOp, multi-op body                  → "elementwise_chain"
//   7. fallback                                          → op->getName().getStringRef()
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/Operation.h"
#include <string>

namespace mlir::auto_fuse {

std::string classifyOpRole(mlir::Operation *op);

} // namespace mlir::auto_fuse
