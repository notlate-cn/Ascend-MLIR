#pragma once

#include "Runtime/Mix/MatmulTilingBackend.h"

namespace mlir::runtime {

struct MixTilingRequest;

MatmulTilingRequest buildMatmulApiTilingRequest(const MixTilingRequest &request);
llvm::Expected<MatmulTilingResult>
generateMatmulApiTiling(const MatmulTilingRequest &request);

class MatmulApiTilingBackend final : public MatmulTilingBackend {
public:
  llvm::StringRef name() const override;
  bool supports(const MatmulTilingRequest &request) const override;
  llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &request) const override;
};

} // namespace mlir::runtime
