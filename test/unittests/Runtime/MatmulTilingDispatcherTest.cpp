#include "Runtime/Mix/MatmulTilingBackend.h"
#include "Runtime/Mix/MatmulTilingDispatcher.h"

#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <utility>

using namespace mlir::runtime;

namespace {

class FakeBackendError final : public llvm::ErrorInfo<FakeBackendError> {
public:
  explicit FakeBackendError(std::string message)
      : message_(std::move(message)) {}

  void log(llvm::raw_ostream &os) const override { os << message_; }

  std::error_code convertToErrorCode() const override {
    return std::make_error_code(std::errc::invalid_argument);
  }

  static char ID;

private:
  std::string message_;
};

char FakeBackendError::ID = 0;

class FakeBackend final : public MatmulTilingBackend {
public:
  FakeBackend(std::string backendName, bool supported,
              std::optional<MatmulTilingResult> result = std::nullopt,
              std::string errorMessage = {})
      : backendName_(std::move(backendName)), supported_(supported),
        result_(std::move(result)), errorMessage_(std::move(errorMessage)) {}

  llvm::StringRef name() const override { return backendName_; }

  bool supports(const MatmulTilingRequest &) const override {
    return supported_;
  }

  llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &) const override {
    if (result_)
      return *result_;
    return llvm::make_error<FakeBackendError>(errorMessage_);
  }

private:
  std::string backendName_;
  bool supported_;
  std::optional<MatmulTilingResult> result_;
  std::string errorMessage_;
};

MatmulTilingResult makeResult(llvm::StringRef backendKind, uint32_t blockDim) {
  MatmulTilingResult result;
  result.backendKind = backendKind.str();
  result.strategyName = "fake";
  result.blockDim = blockDim;
  result.tilingData = {1, 2, 3};
  return result;
}

} // namespace

TEST(MatmulTilingDispatcherTest, PrefersNativeBackendWhenItSucceeds) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";

  FakeBackend native("native", true, makeResult("native", 8));
  FakeBackend api("api", true, makeResult("api", 4));

  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "native");
  EXPECT_EQ(result->blockDim, 8u);
}

TEST(MatmulTilingDispatcherTest, FallsBackToApiWhenNativeErrors) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";

  FakeBackend native("native", true, std::nullopt, "native backend failed");
  FakeBackend api("api", true, makeResult("api", 4));

  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "api");
  EXPECT_EQ(result->blockDim, 4u);
}

TEST(MatmulTilingDispatcherTest, ErrorsWhenApiDoesNotSupportRequest) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";

  FakeBackend native("native", true, std::nullopt, "native backend failed");
  FakeBackend api("api", false, makeResult("api", 4));

  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
}
