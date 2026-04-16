#pragma once

#include "Runtime/Mix/MatmulTilingTypes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

class MatmulTilingBackend {
public:
  virtual ~MatmulTilingBackend() = default;

  virtual llvm::StringRef name() const = 0;
  virtual bool supports(const MatmulTilingRequest &request) const = 0;
  virtual llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &request) const = 0;
};

} // namespace mlir::runtime
