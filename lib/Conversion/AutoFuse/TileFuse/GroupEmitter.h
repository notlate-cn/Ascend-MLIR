#pragma once
#include "Conversion/AutoFuse/GroupInfo.h"
#include "Conversion/AutoFuse/TilePlan.h"
#include "LoopNestBuilder.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

llvm::SmallVector<mlir::Value>
emitGroup(mlir::OpBuilder &builder,
          mlir::Location loc,
          const mlir::auto_fuse::CollapsedGroupInfo &info,
          const mlir::auto_fuse::TilePlan &plan,
          const LoopNestResult &loopNest);

} // namespace mlir::afir
