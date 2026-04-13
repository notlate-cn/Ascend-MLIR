// include/Runtime/ExecutionBackend.h
#pragma once

#include "Runtime/Execution/TaskGraph.h"
#include "Runtime/Profile/ProfileTrace.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace mlir::runtime {

struct ExecutionRequest {
  std::string sessionId;
  RuntimeTask task;
  std::string workingDirectory;
};

struct ExecutionResult {
  std::string taskId;
  std::vector<std::string> producedFiles;
  std::optional<ProfileTrace> profileTrace;
};

class ExecutionBackendDriver {
public:
  virtual ~ExecutionBackendDriver() = default;
  virtual llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) = 0;
};

class ExecutionBackend {
public:
  virtual ~ExecutionBackend() = default;

  virtual ExecutionBackendKind kind() const = 0;
  virtual llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) = 0;
};

llvm::Expected<std::unique_ptr<ExecutionBackend>>
createExecutionBackend(ExecutionBackendKind kind,
                       std::shared_ptr<ExecutionBackendDriver> driver = {});

} // namespace mlir::runtime
