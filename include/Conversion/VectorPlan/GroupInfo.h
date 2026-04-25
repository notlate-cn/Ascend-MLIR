#pragma once
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vector_plan {

enum class AxisRole : uint8_t {
  Parallel,
  Reduction,
};

struct AxisInfo {
  llvm::StringRef name;
  int64_t         staticSize; // ShapedType::kDynamic if dynamic
  AxisRole        role;
};

struct GroupInfo {
  enum class Kind : uint8_t { Vector, Cube };
  Kind                                kind;
  llvm::SmallVector<linalg::LinalgOp> topoMembers;
  llvm::SmallVector<linalg::LinalgOp> sinks;
  llvm::SmallVector<AxisInfo>         canonicalAxes;
  llvm::SmallVector<Value>            boundaryIn;
  llvm::SmallVector<Value>            boundaryOut;
};

struct CollapsedGroupInfo : GroupInfo {
  llvm::SmallVector<AxisInfo> collapsedAxes;
  llvm::SmallVector<int>      axisMap;   // original axis idx -> collapsed axis idx; -1 if not collapsed
  bool                        hasB2      = false;
  bool                        noCollapse = false;
};

struct CubeGroupInfo : GroupInfo {
  linalg::LinalgOp matmul;
  linalg::LinalgOp epilogueAnchor;
};

} // namespace mlir::vector_plan
