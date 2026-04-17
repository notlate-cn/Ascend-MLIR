#pragma once

#include "Runtime/Execution/SessionHandle.h"
#include "Runtime/Execution/TaskGraph.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <map>
#include <string>

namespace mlir::runtime {

class GlobalScheduler {
public:
  llvm::Expected<SessionHandle> submit(ExecutionBackendKind backendKind,
                                       const TaskGraph &graph);

  size_t sessionCount() const { return sessions_.size(); }

private:
  size_t nextSessionOrdinal_ = 0;
  std::map<std::string, ExecutionBackendKind> sessions_;
};

} // namespace mlir::runtime
