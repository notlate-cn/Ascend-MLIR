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
                                      llvm::StringRef artifactPath,
                                      std::optional<int64_t> score = std::nullopt,
                                      std::optional<int64_t> cycleCount = std::nullopt);

void addProfileArtifact(ProfileTrace &trace, llvm::StringRef taskId,
                        ExecutionBackendKind backend,
                        llvm::StringRef artifactPath,
                        std::optional<int64_t> score = std::nullopt,
                        std::optional<int64_t> cycleCount = std::nullopt);

std::string retainedProfileSessionDirectory(llvm::StringRef destinationRoot,
                                            llvm::StringRef sessionId);

std::string retainedProfileSessionSummaryPath(llvm::StringRef destinationRoot,
                                              llvm::StringRef sessionId);

llvm::Expected<ProfileTrace>
retainProfileArtifactsForCli(const ProfileTrace &trace,
                             llvm::StringRef destinationRoot);

llvm::Error pruneRetainedProfileDirectories(llvm::StringRef root,
                                            size_t keepCount);

llvm::Error pruneRetainedProfileDirectoriesForTest(llvm::StringRef root,
                                                   size_t keepCount);

} // namespace mlir::runtime
