#pragma once

#include "Runtime/ProfileTrace.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace mlir::runtime {

bool isSimulatorProfileArtifact(llvm::StringRef path);

std::optional<ProfileTrace>
normalizeSimulatorProfileTrace(llvm::StringRef sessionId,
                               llvm::StringRef taskId,
                               llvm::ArrayRef<std::string> producedFiles);

ProfileEvent makeProfileArtifactEvent(llvm::StringRef taskId,
                                      ExecutionBackendKind backend,
                                      llvm::StringRef artifactPath);

void addProfileArtifact(ProfileTrace &trace, llvm::StringRef taskId,
                        ExecutionBackendKind backend,
                        llvm::StringRef artifactPath);

llvm::Expected<ProfileTrace>
retainProfileArtifactsForCli(const ProfileTrace &trace,
                             llvm::StringRef destinationRoot);

} // namespace mlir::runtime
