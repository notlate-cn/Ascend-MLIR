// include/Runtime/TaskGraph.h
#pragma once

#include "Runtime/Types.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace mlir::runtime {

enum class KernelKind { Vec, Cube, Mix };
enum class MixResourceType { Unknown, AIVOnly, AICOnly, Mix1C1V, Mix1C2V };
enum class ExecutionBackendKind { Simulation, Npu };
enum class BindingSourceKind { ExternalFile, TaskOutput };

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

struct TensorBinding {
  std::string name;
  BindingSourceKind sourceKind = BindingSourceKind::ExternalFile;
  std::string path;
  std::string upstreamTaskId;
  std::string upstreamOutputName;
  std::optional<std::vector<int64_t>> shape;
  std::optional<DType> dtype;
};

struct TilingBinding {
  std::string schemaPath;
  std::string params;
  std::string binaryPath;
};

struct ExecutionInvocation {
  std::vector<TensorBinding> inputs;
  std::vector<TensorBinding> outputs;
  std::vector<TensorBinding> expectedOutputs;
  std::optional<TilingBinding> tiling;
  int blockDim = 1;
  size_t workspaceSize = 8192;
  bool enableProfiling = false;
};

struct RuntimeTask {
  std::string taskId;
  KernelArtifact artifact;
  std::vector<std::string> dependencies;
  ExecutionInvocation invocation;
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
