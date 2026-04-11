// lib/Runtime/ExecutionSession.cpp
#include "Runtime/ExecutionSession.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <filesystem>
#include <deque>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace mlir::runtime {

namespace {

struct SessionRuntimePaths {
  std::string sessionId;
  std::string workingDirectory;
};

struct SchedulerState {
  std::vector<std::string> orderedTaskIds;
  std::vector<std::string> readyTaskIds;
  size_t blockedTaskCount = 0;
  std::map<std::string, RuntimeTask> tasksById;
  std::map<std::string, size_t> remainingDependencies;
  std::map<std::string, std::vector<std::string>> dependents;
  std::deque<std::string> readyQueue;
};

using ProducedBindingMap = std::map<std::string, TensorBinding>;

static std::string bindingKey(llvm::StringRef taskId, llvm::StringRef outputName) {
  std::string key = taskId.str();
  key += "::";
  key += outputName.str();
  return key;
}

static std::string defaultBindingName(size_t index) {
  return "output" + std::to_string(index);
}

static std::string materializeOutputPath(llvm::StringRef workingDirectory,
                                         llvm::StringRef taskId,
                                         llvm::StringRef bindingName,
                                         size_t index) {
  llvm::SmallString<256> path(workingDirectory);
  std::string fileName = taskId.str();
  fileName += "__";
  fileName += bindingName.empty() ? defaultBindingName(index) : bindingName.str();
  fileName += ".npy";
  llvm::sys::path::append(path, fileName);
  return path.str().str();
}

static llvm::Expected<RuntimeTask>
resolveTaskBindings(const RuntimeTask &task, llvm::StringRef workingDirectory,
                    const ProducedBindingMap &producedBindings) {
  RuntimeTask resolved = task;

  for (TensorBinding &binding : resolved.invocation.inputs) {
    if (binding.sourceKind == BindingSourceKind::ExternalFile) {
      if (binding.path.empty()) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "external input binding is missing path: %s",
                                       binding.name.c_str());
      }
      continue;
    }

    if (binding.sourceKind != BindingSourceKind::TaskOutput) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unsupported input binding source for task %s",
                                     task.taskId.c_str());
    }

    const std::string key =
        bindingKey(binding.upstreamTaskId, binding.upstreamOutputName);
    auto it = producedBindings.find(key);
    if (it == producedBindings.end()) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "task input binding cannot resolve upstream output %s for task %s",
          key.c_str(), task.taskId.c_str());
    }

    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = it->second.path;
    if (!binding.shape && it->second.shape)
      binding.shape = it->second.shape;
    if (!binding.dtype && it->second.dtype)
      binding.dtype = it->second.dtype;
  }

  for (size_t index = 0; index < resolved.invocation.outputs.size(); ++index) {
    TensorBinding &binding = resolved.invocation.outputs[index];
    binding.sourceKind = BindingSourceKind::ExternalFile;
    if (binding.path.empty()) {
      binding.path = materializeOutputPath(workingDirectory, resolved.taskId,
                                           binding.name, index);
    }
  }

  return resolved;
}

static void recordProducedBindings(const RuntimeTask &task,
                                   ProducedBindingMap &producedBindings) {
  for (size_t index = 0; index < task.invocation.outputs.size(); ++index) {
    TensorBinding binding = task.invocation.outputs[index];
    if (binding.name.empty())
      binding.name = defaultBindingName(index);
    producedBindings[bindingKey(task.taskId, binding.name)] = std::move(binding);
  }
}

static llvm::Expected<SessionRuntimePaths> prepareSessionRuntimePaths() {
  std::error_code tempDirError;
  const std::filesystem::path sessionRoot =
      std::filesystem::temp_directory_path(tempDirError) / "ascendc-runtime";
  if (tempDirError)
    return llvm::createStringError(
        tempDirError, "Cannot determine temp directory for runtime sessions");

  if (auto ec = llvm::sys::fs::create_directories(sessionRoot.string()))
    return llvm::createStringError(ec, "Cannot create session root directory: %s",
                                   sessionRoot.string().c_str());

  llvm::SmallString<256> workingDirectory;
  if (auto ec = llvm::sys::fs::createUniqueDirectory(
          (sessionRoot.string() + "/runtime-session-"), workingDirectory))
    return llvm::createStringError(ec, "Cannot create session directory in %s",
                                   sessionRoot.string().c_str());

  return SessionRuntimePaths{
      llvm::sys::path::filename(workingDirectory).str(),
      workingDirectory.str().str(),
  };
}

static llvm::Expected<SchedulerState> buildSchedulerState(const TaskGraph &graph) {
  auto orderedTasksOr = graph.executionOrder();
  if (!orderedTasksOr)
    return orderedTasksOr.takeError();

  SchedulerState state;
  state.orderedTaskIds.reserve(orderedTasksOr->size());

  for (const RuntimeTask &task : *orderedTasksOr) {
    state.orderedTaskIds.push_back(task.taskId);
    state.tasksById.emplace(task.taskId, task);
    state.remainingDependencies.emplace(task.taskId, task.dependencies.size());
    for (const std::string &dependency : task.dependencies)
      state.dependents[dependency].push_back(task.taskId);
  }

  for (const RuntimeTask &task : *orderedTasksOr) {
    auto it = state.remainingDependencies.find(task.taskId);
    if (it == state.remainingDependencies.end())
      continue;
    if (it->second == 0) {
      state.readyTaskIds.push_back(task.taskId);
      state.readyQueue.push_back(task.taskId);
    } else {
      ++state.blockedTaskCount;
    }
  }

  return state;
}

static llvm::Error canScheduleTask(const RuntimeTask &) {
  return llvm::Error::success();
}

} // namespace

ExecutionSession::ExecutionSession(ExecutionBackendKind backendKind)
    : backendKind_(backendKind) {}

ExecutionSession::ExecutionSession(ExecutionBackendKind backendKind,
                                   std::shared_ptr<ExecutionBackendDriver> driver)
    : backendKind_(backendKind), driver_(std::move(driver)) {}

ExecutionSession::~ExecutionSession() {
  for (const std::string &workingDirectory : workingDirectories_) {
    std::error_code ec;
    std::filesystem::remove_all(workingDirectory, ec);
  }
}

llvm::Expected<SessionPlan> ExecutionSession::plan(const TaskGraph &graph) const {
  auto schedulerOr = buildSchedulerState(graph);
  if (!schedulerOr)
    return schedulerOr.takeError();
  return SessionPlan{std::move(schedulerOr->orderedTaskIds),
                     std::move(schedulerOr->readyTaskIds),
                     schedulerOr->blockedTaskCount};
}

llvm::Expected<ProfileTrace> ExecutionSession::run(const TaskGraph &graph) {
  auto schedulerOr = buildSchedulerState(graph);
  if (!schedulerOr)
    return schedulerOr.takeError();
  SchedulerState scheduler = std::move(*schedulerOr);

  auto backendOr = getOrCreateBackend();
  if (!backendOr)
    return backendOr.takeError();
  ExecutionBackend &backend = *backendOr;

  auto runtimePathsOr = prepareSessionRuntimePaths();
  if (!runtimePathsOr)
    return runtimePathsOr.takeError();

  ProfileTrace sessionTrace;
  sessionTrace.sessionId = runtimePathsOr->sessionId;
  const std::string &workingDirectory = runtimePathsOr->workingDirectory;
  workingDirectories_.push_back(workingDirectory);
  ProducedBindingMap producedBindings;
  std::set<std::string> completedTasks;

  while (!scheduler.readyQueue.empty()) {
    const std::string taskId = scheduler.readyQueue.front();
    scheduler.readyQueue.pop_front();
    auto taskIt = scheduler.tasksById.find(taskId);
    if (taskIt == scheduler.tasksById.end()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "scheduler lost task definition for %s",
                                     taskId.c_str());
    }
    const RuntimeTask &task = taskIt->second;
    if (auto err = canScheduleTask(task))
      return std::move(err);
    auto resolvedTaskOr =
        resolveTaskBindings(task, workingDirectory, producedBindings);
    if (!resolvedTaskOr)
      return resolvedTaskOr.takeError();

    ExecutionRequest request;
    request.sessionId = sessionTrace.sessionId;
    request.task = std::move(*resolvedTaskOr);
    request.workingDirectory = workingDirectory;

    auto resultOr = backend.run(request);
    if (!resultOr)
      return resultOr.takeError();

    recordProducedBindings(request.task, producedBindings);
    completedTasks.insert(taskId);

    if (resultOr->profileTrace) {
      for (ProfileEvent event : resultOr->profileTrace->events)
        sessionTrace.addEvent(std::move(event));
    }

    auto dependentsIt = scheduler.dependents.find(taskId);
    if (dependentsIt == scheduler.dependents.end())
      continue;
    for (const std::string &dependentId : dependentsIt->second) {
      auto depCountIt = scheduler.remainingDependencies.find(dependentId);
      if (depCountIt == scheduler.remainingDependencies.end())
        continue;
      if (depCountIt->second == 0)
        continue;
      --depCountIt->second;
      if (depCountIt->second == 0)
        scheduler.readyQueue.push_back(dependentId);
    }
  }

  if (completedTasks.size() != scheduler.orderedTaskIds.size()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "scheduler terminated with %zu/%zu tasks completed",
        completedTasks.size(), scheduler.orderedTaskIds.size());
  }

  return sessionTrace;
}

llvm::Expected<ExecutionBackend &> ExecutionSession::getOrCreateBackend() {
  if (!backend_) {
    auto backendOr = createExecutionBackend(backendKind_, driver_);
    if (!backendOr)
      return backendOr.takeError();
    backend_ = std::move(*backendOr);
  }
  return *backend_;
}

} // namespace mlir::runtime
