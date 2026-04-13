// include/Runtime/ExecutionSession.h
#pragma once

#include "Runtime/Execution/ExecutionBackend.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::runtime {

struct SessionPlan {
  std::vector<std::string> orderedTaskIds;
  std::vector<std::string> readyTaskIds;
  size_t blockedTaskCount = 0;
};

class ExecutionSession {
public:
  explicit ExecutionSession(ExecutionBackendKind backendKind);
  ExecutionSession(ExecutionBackendKind backendKind,
                   std::shared_ptr<ExecutionBackendDriver> driver);
  ~ExecutionSession();

  llvm::Expected<SessionPlan> plan(const TaskGraph &graph) const;
  llvm::Expected<ProfileTrace> run(const TaskGraph &graph);

private:
  llvm::Expected<ExecutionBackend &> getOrCreateBackend();

  ExecutionBackendKind backendKind_;
  std::shared_ptr<ExecutionBackendDriver> driver_;
  std::unique_ptr<ExecutionBackend> backend_;
  std::vector<std::string> workingDirectories_;
};

} // namespace mlir::runtime
