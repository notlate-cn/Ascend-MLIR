//===- TilingRealizationDriver.h - Ascend tiling realization ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TILINGREALIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TILINGREALIZATIONDRIVER_H

#include "RealizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::ascend::realize {

/// Drives Schedule-decision-based tiling of linalg ops in a module using
/// scf::tileUsingSCF. Operates on tensor IR (pre-bufferize).
class TilingRealizationDriver {
public:
  /// Tile every linalg.generic / linalg.matmul that carries
  /// ascend.schedule.selected_tile_shape. Replaces the op in-place with a
  /// scf.for nest. Returns failure if any tile shape is malformed.
  LogicalResult tileModule(ModuleOp module) const;
};

} // namespace mlir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TILINGREALIZATIONDRIVER_H
