// include/Runtime/TaskGraph.h
#pragma once

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace mlir::runtime {

enum class KernelKind { Vec, Cube, Mix };
enum class MixResourceType { Unknown, AIVOnly, AICOnly, Mix1C1V, Mix1C2V };
enum class ExecutionBackendKind { Simulation, Npu };

struct KernelArtifact {
  std::string kernelName;
  KernelKind kernelKind = KernelKind::Vec;
  MixResourceType mixResourceType = MixResourceType::Unknown;
  std::string socVersion;
  std::string artifactRoot;
  std::string deviceBinaryPath;
  std::string packedSharedObjectPath;
  std::string manifestPath;
};

struct RuntimeTask {
  std::string taskId;
  KernelArtifact artifact;
  std::vector<std::string> dependencies;
};

class TaskGraph {
public:
  llvm::Error addTask(const RuntimeTask &task);
  llvm::Expected<std::vector<RuntimeTask>> executionOrder() const;
  llvm::Expected<std::vector<RuntimeTask>> orderedTasks() const;
  llvm::Expected<std::vector<std::string>> topologicalOrder() const;

private:
  std::vector<RuntimeTask> tasks_;
};

} // namespace mlir::runtime
