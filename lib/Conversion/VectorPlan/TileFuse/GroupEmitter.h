#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "LoopNestBuilder.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

llvm::SmallVector<mlir::Value>
emitGroup(mlir::OpBuilder &builder,
          mlir::Location loc,
          const mlir::vector_plan::CollapsedGroupInfo &info,
          const mlir::vector_plan::TilePlan &plan,
          const LoopNestResult &loopNest);

} // namespace mlir::afir
