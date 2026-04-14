#pragma once

#include "Runtime/Execution/ExecutionRunner.h"

#include <memory>

namespace mlir::runtime {

llvm::Expected<std::unique_ptr<ExecutionRunner>>
createDefaultExecutionRunner(ExecutionRunnerMode mode);

} // namespace mlir::runtime
