#include "Runtime/Mix/MatmulTilingBackend.h"
#include "Runtime/Mix/MatmulTilingDispatcher.h"

#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

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
    ++supportsCalls_;
    return supported_;
  }

  llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &) const override {
    ++generateCalls_;
    if (result_)
      return *result_;
    return llvm::make_error<FakeBackendError>(errorMessage_);
  }

  unsigned supportsCalls() const { return supportsCalls_; }
  unsigned generateCalls() const { return generateCalls_; }

private:
  std::string backendName_;
  bool supported_;
  std::optional<MatmulTilingResult> result_;
  std::string errorMessage_;
  mutable unsigned supportsCalls_ = 0;
  mutable unsigned generateCalls_ = 0;
};

MatmulTilingResult makeResult(llvm::StringRef backendKind, uint32_t blockDim) {
  MatmulTilingResult result;
  result.backendKind = backendKind.str();
  result.strategyName = "fake";
  result.blockDim = blockDim;
  result.tilingData = {1, 2, 3};
  return result;
}

std::vector<std::string> collectErrorMessages(llvm::Error err) {
  std::vector<std::string> messages;
  llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase &info) {
    messages.push_back(info.message());
  });
  return messages;
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
  EXPECT_EQ(api.generateCalls(), 0u);
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
  EXPECT_EQ(native.generateCalls(), 1u);
  EXPECT_EQ(api.generateCalls(), 1u);
}

TEST(MatmulTilingDispatcherTest, PreservesNativeFailureWhenApiDoesNotSupportRequest) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";

  FakeBackend native("native", true, std::nullopt, "native backend failed");
  FakeBackend api("api", false, makeResult("api", 4));

  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_FALSE(static_cast<bool>(result));
  auto messages = collectErrorMessages(result.takeError());
  ASSERT_EQ(messages.size(), 2u);
  EXPECT_EQ(messages[0], "native backend failed");
  EXPECT_NE(messages[1].find("native matmul tiling backend native claimed support for kernel matmul_bias_relu but failed"), std::string::npos);
  EXPECT_NE(messages[1].find("api backend api does not support the request"), std::string::npos);
  EXPECT_EQ(native.generateCalls(), 1u);
  EXPECT_EQ(api.generateCalls(), 0u);
}

TEST(MatmulTilingDispatcherTest, PreservesNativeAndApiFailures) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";

  FakeBackend native("native", true, std::nullopt, "native backend failed");
  FakeBackend api("api", true, std::nullopt, "api backend failed");

  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_FALSE(static_cast<bool>(result));
  auto messages = collectErrorMessages(result.takeError());
  ASSERT_EQ(messages.size(), 3u);
  EXPECT_EQ(messages[0], "native backend failed");
  EXPECT_EQ(messages[1], "api backend failed");
  EXPECT_NE(messages[2].find("native matmul tiling backend native claimed support for kernel matmul_bias_relu but failed"), std::string::npos);
  EXPECT_NE(messages[2].find("api backend api also failed"), std::string::npos);
}

TEST(MatmulTilingDispatcherTest, NativeUnsupportedApiFailureIsReturned) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";

  FakeBackend native("native", false, makeResult("native", 8));
  FakeBackend api("api", true, std::nullopt, "api backend failed");

  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_FALSE(static_cast<bool>(result));
  auto messages = collectErrorMessages(result.takeError());
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages[0], "api backend failed");
  EXPECT_EQ(native.generateCalls(), 0u);
  EXPECT_EQ(api.generateCalls(), 1u);
}

TEST(MatmulTilingDispatcherTest, BothBackendsUnsupported) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";

  FakeBackend native("native", false, makeResult("native", 8));
  FakeBackend api("api", false, makeResult("api", 4));

  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_FALSE(static_cast<bool>(result));
  auto messages = collectErrorMessages(result.takeError());
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages[0],
            "no matmul tiling backend supports kernel matmul_bias_relu");
  EXPECT_EQ(native.generateCalls(), 0u);
  EXPECT_EQ(api.generateCalls(), 0u);
}
