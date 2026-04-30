// lib/Runtime/TaskGraph.cpp
#include "Runtime/TaskGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <queue>
#include <unordered_map>

using namespace mlir::runtime;

llvm::Error TaskGraph::addTask(const RuntimeTask &task) {
  if (task.taskId.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "empty task id");
  }
  auto it = llvm::find_if(tasks_, [&](const RuntimeTask &existing) {
    return existing.taskId == task.taskId;
  });
  if (it != tasks_.end()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "duplicate task id");
  }
  tasks_.push_back(task);
  return llvm::Error::success();
}

llvm::Expected<std::vector<RuntimeTask>> TaskGraph::executionOrder() const {
  return orderedTasks();
}

llvm::Expected<std::vector<RuntimeTask>> TaskGraph::orderedTasks() const {
  std::unordered_map<std::string, size_t> indexById;
  indexById.reserve(tasks_.size());
  for (size_t i = 0; i < tasks_.size(); ++i)
    indexById.emplace(tasks_[i].taskId, i);

  std::vector<size_t> indegree(tasks_.size(), 0);
  std::vector<std::vector<size_t>> outgoing(tasks_.size());

  for (size_t i = 0; i < tasks_.size(); ++i) {
    const auto &task = tasks_[i];
    for (const auto &dep : task.dependencies) {
      auto depIt = indexById.find(dep);
      if (depIt == indexById.end()) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "unknown task dependency");
      }
      outgoing[depIt->second].push_back(i);
      ++indegree[i];
    }
  }

  std::queue<size_t> ready;
  for (size_t i = 0; i < tasks_.size(); ++i) {
    if (indegree[i] == 0)
      ready.push(i);
  }

  std::vector<RuntimeTask> order;
  order.reserve(tasks_.size());
  while (!ready.empty()) {
    size_t i = ready.front();
    ready.pop();
    order.push_back(tasks_[i]);

    for (size_t next : outgoing[i]) {
      if (--indegree[next] == 0)
        ready.push(next);
    }
  }

  if (order.size() != tasks_.size()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "task graph contains a cycle");
  }

  return order;
}

llvm::Expected<std::vector<std::string>> TaskGraph::topologicalOrder() const {
  std::vector<std::string> order;
  auto orderedOr = orderedTasks();
  if (!orderedOr)
    return orderedOr.takeError();
  order.reserve(orderedOr->size());
  for (const RuntimeTask &task : *orderedOr)
    order.push_back(task.taskId);
  return order;
}
