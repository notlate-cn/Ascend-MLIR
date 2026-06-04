//===- SymbolAxisSpace.h - Ascend symbolic axis space ----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_SYMBOLAXISSPACE_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_SYMBOLAXISSPACE_H

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::ascend::kernelize {

struct LogicalAxis {
  Operation *func = nullptr;
  unsigned axisId = 0;
  std::string symbolName;
  IteratorKind kind = IteratorKind::Unknown;
  bool hasMixedIteratorKinds = false;
  unsigned memberCount = 0;
};

struct FunctionAxisSpace {
  Operation *func = nullptr;
  SmallVector<LogicalAxis, 4> axes;
};

struct OpAxisRef {
  int64_t axisId = -1;
  std::string symbolName;
  IteratorKind iteratorKind = IteratorKind::Unknown;

  bool hasAxis() const { return axisId >= 0; }
};

struct SymbolAxisSpace {
  FunctionAxisSpace function;
  DenseMap<Operation *, SmallVector<OpAxisRef, 4>> opAxisMap;
};

FailureOr<SymbolAxisSpace>
buildSymbolAxisSpace(func::FuncOp func, ArrayRef<Operation *> ops);

FailureOr<SymbolAxisSpace> buildSymbolAxisSpace(func::FuncOp func);

} // namespace mlir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_SYMBOLAXISSPACE_H
