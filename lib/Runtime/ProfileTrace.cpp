// lib/Runtime/ProfileTrace.cpp
#include "Runtime/ProfileTrace.h"

#include <utility>

namespace mlir::runtime {

void ProfileTrace::addEvent(ProfileEvent event) {
  events.push_back(std::move(event));
}

void ProfileTrace::addProfileArtifact(llvm::StringRef taskId,
                                      ExecutionBackendKind backend,
                                      llvm::StringRef artifactPath) {
  addEvent(ProfileEvent{taskId.str(), backend, "profile_artifact",
                        artifactPath.str()});
}

} // namespace mlir::runtime
