#pragma once

#include "Runtime/Mix/MatmulTilingBackend.h"

namespace mlir::runtime {

llvm::Expected<MatmulTilingResult>
dispatchMatmulTiling(const MatmulTilingRequest &request,
                     const MatmulTilingBackend &nativeBackend,
                     const MatmulTilingBackend &apiBackend);

} // namespace mlir::runtime
