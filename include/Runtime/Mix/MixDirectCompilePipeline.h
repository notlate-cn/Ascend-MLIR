#pragma once

#include "Runtime/Mix/MixArtifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::Expected<MixArtifact>
executeMixDirectCompile(llvm::StringRef outputDir, llvm::StringRef kernelSrc,
                        llvm::StringRef kernelName,
                        llvm::StringRef cannMlirPath, llvm::StringRef npyDir,
                        llvm::StringRef socVersion);

} // namespace mlir::runtime
