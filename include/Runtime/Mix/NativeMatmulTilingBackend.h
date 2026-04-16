#pragma once

#include "Runtime/Mix/MatmulTilingBackend.h"

namespace mlir::runtime {

class NativeMatmulTilingBackend final : public MatmulTilingBackend {
public:
  llvm::StringRef name() const override;
  bool supports(const MatmulTilingRequest &request) const override;
  llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &request) const override;
};

} // namespace mlir::runtime

