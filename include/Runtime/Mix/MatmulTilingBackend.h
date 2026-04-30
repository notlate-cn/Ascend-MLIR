#pragma once

#include "Runtime/Mix/MatmulTilingTypes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

class MatmulTilingBackend {
public:
  virtual ~MatmulTilingBackend() = default;

  virtual llvm::StringRef name() const = 0;
  // `supports()` should be a cheap capability check for the same request
  // passed to `generate()`. If it returns true, `generate()` is expected to
  // either produce a result for that request or return a detailed `llvm::Error`
  // explaining why the backend could not finish.
  virtual bool supports(const MatmulTilingRequest &request) const = 0;
  virtual llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &request) const = 0;
};

} // namespace mlir::runtime
