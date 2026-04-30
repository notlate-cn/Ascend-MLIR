#pragma once

#include "Runtime/Mix/MixAbi.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::Expected<MixAbiMetadata>
extractMixAbiFromCannMlir(llvm::StringRef cannMlirPath);

} // namespace mlir::runtime
