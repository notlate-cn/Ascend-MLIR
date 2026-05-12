#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::afir {

mlir::Value castToIndex(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v);

mlir::Value getAxisExtentValue(mlir::OpBuilder &b, mlir::Location loc,
                                const mlir::vector_plan::CollapsedGroupInfo &info,
                                int axisIdx);

// True when every use of every result of `op` is by another member of the
// group — i.e. `op`'s result is an intra-group intermediate (produced and
// consumed inside the same tile-fuse loop body), not a group output that must
// escape via an scf.for iter_arg.  Such a result should get a fresh on-chip
// (VECCALC) tile per iteration instead of being accumulated into a full-shape
// tensor that bufferizes to a GM buffer.  (A linalg.transpose with >1 consumer
// is the canonical case — `--linalg-fuse-elementwise-ops` cannot absorb it.)
bool resultUsedOnlyByGroupMembers(
    mlir::linalg::LinalgOp op,
    const mlir::vector_plan::CollapsedGroupInfo &info);

} // namespace mlir::afir
