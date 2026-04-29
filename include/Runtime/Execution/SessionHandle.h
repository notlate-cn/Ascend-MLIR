#pragma once

#include <string>
#include <utility>

namespace mlir::runtime {

class SessionHandle {
public:
  explicit SessionHandle(std::string sessionId)
      : sessionId_(std::move(sessionId)) {}

  const std::string &sessionId() const { return sessionId_; }

private:
  std::string sessionId_;
};

} // namespace mlir::runtime
