// lib/Runtime/ExecutionSession.cpp
#include "Runtime/ExecutionSession.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <thread>
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

static bool forceSerialSchedulerFromEnv() {
  const char *value = std::getenv("ASCEND_RUNTIME_FORCE_SERIAL_SCHEDULER");
  if (!value)
    return false;
  return value[0] != '\0' && std::string_view(value) != "0";
}

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

static const char *mixResourceTypeToString(MixResourceType type) {
  switch (type) {
  case MixResourceType::Unknown:
    return "unknown";
  case MixResourceType::AIVOnly:
    return "aiv_only";
  case MixResourceType::AICOnly:
    return "aic_only";
  case MixResourceType::Mix1C1V:
    return "mix_1c1v";
  case MixResourceType::Mix1C2V:
    return "mix_1c2v";
  }
  return "unknown";
}

static llvm::Error canScheduleTask(const RuntimeTask &task) {
  switch (task.artifact.kernelKind) {
  case KernelKind::Vec:
  case KernelKind::Cube:
    return llvm::Error::success();
  case KernelKind::Mix:
    switch (task.artifact.mixResourceType) {
    case MixResourceType::Mix1C1V:
    case MixResourceType::Mix1C2V:
      return llvm::Error::success();
    case MixResourceType::Unknown:
    case MixResourceType::AIVOnly:
    case MixResourceType::AICOnly:
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "task %s requests unsupported mix resource type: %s",
          task.taskId.c_str(),
          mixResourceTypeToString(task.artifact.mixResourceType));
    }
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "task %s requests unsupported mix resource type: %s",
        task.taskId.c_str(),
        mixResourceTypeToString(task.artifact.mixResourceType));
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "task %s has unsupported kernel kind",
                                 task.taskId.c_str());
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

void ExecutionSession::releaseWorkingDirectoriesForProcessExit() {
  workingDirectories_.clear();
}

GlobalScheduler &ExecutionSession::globalScheduler() {
  static GlobalScheduler scheduler;
  return scheduler;
}

llvm::Expected<SessionHandle> ExecutionSession::submit(const TaskGraph &graph) {
  return globalScheduler().submit(backendKind_, graph);
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

  auto runtimePathsOr = prepareSessionRuntimePaths();
  if (!runtimePathsOr)
    return runtimePathsOr.takeError();

  ProfileTrace sessionTrace;
  sessionTrace.sessionId = runtimePathsOr->sessionId;
  sessionTrace.setAttribute("scheduler_mode", "serial");
  sessionTrace.addCounter("planned_task_count",
                          static_cast<int64_t>(scheduler.orderedTaskIds.size()));
  const std::string &workingDirectory = runtimePathsOr->workingDirectory;
  workingDirectories_.push_back(workingDirectory);
  ProducedBindingMap producedBindings;
  std::set<std::string> completedTasks;

  auto backendOr = getOrCreateBackend();
  if (!backendOr)
    return backendOr.takeError();
  ExecutionBackend &backend = *backendOr;
  const bool enableConcurrentDispatch =
      !forceSerialSchedulerFromEnv() &&
      backend.allowsConcurrentTaskDispatch() &&
      scheduler.orderedTaskIds.size() > 1;

  if (enableConcurrentDispatch) {
    sessionTrace.setAttribute("scheduler_mode", "concurrent");
    sessionTrace.addCounter("frontier_count", 1);
    sessionTrace.counters["max_frontier_width"] =
        static_cast<int64_t>(scheduler.readyQueue.size());
    std::mutex schedulerMutex;
    std::condition_variable schedulerCv;
    bool failed = false;
    size_t inFlightTasks = 0;
    size_t maxInFlightTasks = 0;
    std::string firstErrorMessage;
    std::deque<std::string> nextReadyQueue;

    const unsigned concurrencyHint = std::thread::hardware_concurrency();
    const size_t workerCount =
        std::max<size_t>(1, std::min<size_t>(
                                scheduler.orderedTaskIds.size(),
                                concurrencyHint == 0 ? 4 : concurrencyHint));
    std::vector<std::thread> workers;
    workers.reserve(workerCount);

    auto worker = [&]() {
      ExecutionBackend *workerBackend = &backend;
      std::unique_ptr<ExecutionBackend> ownedBackend;
      if (driver_) {
        auto ownedBackendOr = createExecutionBackend(backendKind_, driver_);
        if (!ownedBackendOr) {
          std::lock_guard<std::mutex> lock(schedulerMutex);
          if (!failed) {
            failed = true;
            firstErrorMessage = llvm::toString(ownedBackendOr.takeError());
          } else {
            llvm::consumeError(ownedBackendOr.takeError());
          }
          schedulerCv.notify_all();
          return;
        }
        ownedBackend = std::move(*ownedBackendOr);
        workerBackend = ownedBackend.get();
      }

      if (!workerBackend) {
        std::lock_guard<std::mutex> lock(schedulerMutex);
        if (!failed) {
          failed = true;
          firstErrorMessage = "scheduler failed to create worker backend";
        }
        schedulerCv.notify_all();
        return;
      }

      while (true) {
        ExecutionRequest request;
        std::string taskId;

        {
          std::unique_lock<std::mutex> lock(schedulerMutex);
          schedulerCv.wait(lock, [&] {
            return failed || !scheduler.readyQueue.empty() || inFlightTasks == 0;
          });

          if (failed) {
            if (inFlightTasks == 0)
              return;
            continue;
          }

          if (scheduler.readyQueue.empty()) {
            if (inFlightTasks == 0) {
              if (!nextReadyQueue.empty()) {
                scheduler.readyQueue.swap(nextReadyQueue);
                sessionTrace.addCounter("frontier_count", 1);
                sessionTrace.counters["max_frontier_width"] = std::max(
                    sessionTrace.counters["max_frontier_width"],
                    static_cast<int64_t>(scheduler.readyQueue.size()));
                schedulerCv.notify_all();
                continue;
              }
              return;
            }
            continue;
          }

          taskId = scheduler.readyQueue.front();
          scheduler.readyQueue.pop_front();

          auto taskIt = scheduler.tasksById.find(taskId);
          if (taskIt == scheduler.tasksById.end()) {
            failed = true;
            firstErrorMessage =
                "scheduler lost task definition for " + taskId;
            schedulerCv.notify_all();
            continue;
          }

          if (auto err = canScheduleTask(taskIt->second)) {
            failed = true;
            firstErrorMessage = llvm::toString(std::move(err));
            schedulerCv.notify_all();
            continue;
          }

          auto resolvedTaskOr = resolveTaskBindings(taskIt->second, workingDirectory,
                                                    producedBindings);
          if (!resolvedTaskOr) {
            failed = true;
            firstErrorMessage = llvm::toString(resolvedTaskOr.takeError());
            schedulerCv.notify_all();
            continue;
          }

          request.sessionId = sessionTrace.sessionId;
          request.task = std::move(*resolvedTaskOr);
          request.workingDirectory = workingDirectory;
          ++inFlightTasks;
          maxInFlightTasks = std::max(maxInFlightTasks, inFlightTasks);
        }

        auto resultOr = workerBackend->run(request);

        {
          std::lock_guard<std::mutex> lock(schedulerMutex);
          --inFlightTasks;

          if (!resultOr) {
            if (!failed) {
              failed = true;
              firstErrorMessage = llvm::toString(resultOr.takeError());
            } else {
              llvm::consumeError(resultOr.takeError());
            }
            schedulerCv.notify_all();
            continue;
          }

          recordProducedBindings(request.task, producedBindings);
          completedTasks.insert(taskId);

          if (resultOr->profileTrace) {
            sessionTrace.merge(*resultOr->profileTrace);
          }

          if (!failed) {
            auto dependentsIt = scheduler.dependents.find(taskId);
            if (dependentsIt != scheduler.dependents.end()) {
              for (const std::string &dependentId : dependentsIt->second) {
                auto depCountIt =
                    scheduler.remainingDependencies.find(dependentId);
                if (depCountIt == scheduler.remainingDependencies.end())
                  continue;
                if (depCountIt->second == 0)
                  continue;
                --depCountIt->second;
                if (depCountIt->second == 0)
                  nextReadyQueue.push_back(dependentId);
              }
            }
          }

          if (!failed && inFlightTasks == 0 && scheduler.readyQueue.empty() &&
              !nextReadyQueue.empty()) {
            scheduler.readyQueue.swap(nextReadyQueue);
            sessionTrace.addCounter("frontier_count", 1);
            sessionTrace.counters["max_frontier_width"] = std::max(
                sessionTrace.counters["max_frontier_width"],
                static_cast<int64_t>(scheduler.readyQueue.size()));
          }

          schedulerCv.notify_all();
        }
      }
    };

    for (size_t i = 0; i < workerCount; ++i)
      workers.emplace_back(worker);
    for (std::thread &thread : workers)
      thread.join();

    sessionTrace.counters["max_in_flight_tasks"] =
        static_cast<int64_t>(maxInFlightTasks);
    if (failed)
      return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                     firstErrorMessage.c_str());
  } else {
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
        sessionTrace.merge(*resultOr->profileTrace);
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
