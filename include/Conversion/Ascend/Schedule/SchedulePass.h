//===- SchedulePass.h - Ascend schedule pass -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::ascend {

std::unique_ptr<Pass> createAscendSchedulePass();

} // namespace mlir::ascend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PASS_H
