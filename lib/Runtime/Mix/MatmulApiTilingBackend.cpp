#include "Runtime/Mix/MatmulApiTilingBackend.h"

#include "llvm/Support/Error.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <limits>
#include <optional>
#include <string>

namespace mlir::runtime {
namespace {

static llvm::Expected<matmul_tiling::DataType> toMatmulDataType(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return matmul_tiling::DataType::DT_FLOAT16;
  case DType::BF16:
    return matmul_tiling::DataType::DT_BF16;
  case DType::F32:
    return matmul_tiling::DataType::DT_FLOAT;
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported matmul api tiling dtype");
  }
}

static const char *toString(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return "F16";
  case DType::BF16:
    return "BF16";
  case DType::F32:
    return "F32";
  case DType::INT8:
    return "INT8";
  case DType::INT32:
    return "INT32";
  case DType::INT64:
    return "INT64";
  }
  return "unknown";
}

static bool isPositiveShape(const MatmulTilingRequest &request) {
  return request.problem.M > 0 && request.problem.N > 0 &&
         request.problem.K > 0;
}

static bool hasSupportedLayout(const MatmulTilingRequest &request) {
  return request.problem.layoutA == MatmulLayout::ND &&
         request.problem.layoutB == MatmulLayout::ND &&
         request.problem.layoutC == MatmulLayout::ND;
}

static bool isSupportedDType(DType dtype) {
  switch (dtype) {
  case DType::F16:
  case DType::BF16:
  case DType::F32:
    return true;
  default:
    return false;
  }
}

static bool hasSupportedDType(const MatmulTilingRequest &request) {
  return isSupportedDType(request.problem.dtypeA) &&
         isSupportedDType(request.problem.dtypeB) &&
         isSupportedDType(request.problem.dtypeC) &&
         (!request.problem.hasBias || !request.problem.biasDType.has_value() ||
          isSupportedDType(*request.problem.biasDType));
}

static const char *traverseToString(std::optional<MatrixTraverseKind> traverse) {
  if (!traverse.has_value())
    return "FIRSTM";
  switch (*traverse) {
  case MatrixTraverseKind::FirstM:
    return "FIRSTM";
  case MatrixTraverseKind::FirstN:
    return "FIRSTN";
  }
  return "FIRSTM";
}

static matmul_tiling::MatrixTraverse
toMatmulTraverse(std::optional<MatrixTraverseKind> traverse) {
  if (!traverse.has_value())
    return matmul_tiling::MatrixTraverse::FIRSTM;
  switch (*traverse) {
  case MatrixTraverseKind::FirstM:
    return matmul_tiling::MatrixTraverse::FIRSTM;
  case MatrixTraverseKind::FirstN:
    return matmul_tiling::MatrixTraverse::FIRSTN;
  }
  return matmul_tiling::MatrixTraverse::FIRSTM;
}

static std::string resolveSocVersion(const MatmulTilingRequest &request) {
  if (!request.hints.socVersion.empty())
    return request.hints.socVersion;
  return "Ascend910B1";
}

static DType resolveBiasDType(const MatmulTilingRequest &request) {
  return request.problem.biasDType.value_or(request.problem.dtypeC);
}

static bool supportsMatmulApiTilingRequest(const MatmulTilingRequest &request) {
  return isPositiveShape(request) && request.problem.batchShape.empty() &&
         hasSupportedLayout(request) && hasSupportedDType(request);
}

static llvm::Expected<int> toVendorInt64RangeChecked(int64_t value,
                                                     llvm::StringRef name) {
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    const std::string nameStr = name.str();
    const std::string valueStr = std::to_string(value);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "matmul api tiling %s=%s is out of range for int", nameStr.c_str(),
        valueStr.c_str());
  }
  return static_cast<int>(value);
}

static llvm::Expected<MatmulTilingResult>
generateMatmulApiTilingImpl(const MatmulTilingRequest &request) {
  if (!supportsMatmulApiTilingRequest(request)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "unsupported matmul api tiling request for kernel %s",
        request.kernelName.c_str());
  }

  auto aDTypeOr = toMatmulDataType(request.problem.dtypeA);
  if (!aDTypeOr)
    return aDTypeOr.takeError();
  auto bDTypeOr = toMatmulDataType(request.problem.dtypeB);
  if (!bDTypeOr)
    return bDTypeOr.takeError();
  auto cDTypeOr = toMatmulDataType(request.problem.dtypeC);
  if (!cDTypeOr)
    return cDTypeOr.takeError();
  std::optional<matmul_tiling::DataType> biasDType;
  if (request.problem.hasBias) {
    auto biasDTypeOr = toMatmulDataType(resolveBiasDType(request));
    if (!biasDTypeOr)
      return biasDTypeOr.takeError();
    biasDType = *biasDTypeOr;
  }

  const std::string socVersion = resolveSocVersion(request);
  auto *ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion.c_str());
  if (!ascendcPlatform) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "cannot initialize AscendC platform for soc %s", socVersion.c_str());
  }

  auto mOr = toVendorInt64RangeChecked(request.problem.M, "M");
  if (!mOr)
    return mOr.takeError();
  auto nOr = toVendorInt64RangeChecked(request.problem.N, "N");
  if (!nOr)
    return nOr.takeError();
  auto kOr = toVendorInt64RangeChecked(request.problem.K, "K");
  if (!kOr)
    return kOr.takeError();

  matmul_tiling::MatmulApiTiling tilingApi(*ascendcPlatform);
  tilingApi.SetAType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND, *aDTypeOr,
                     request.problem.transA);
  tilingApi.SetBType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND, *bDTypeOr,
                     request.problem.transB);
  tilingApi.SetCType(matmul_tiling::TPosition::GM,
                     matmul_tiling::CubeFormat::ND, *cDTypeOr);
  if (biasDType) {
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM,
                          matmul_tiling::CubeFormat::ND, *biasDType);
  }
  tilingApi.SetOrgShape(*mOr, *nOr, *kOr);
  tilingApi.SetShape(*mOr, *nOr, *kOr);
  tilingApi.SetBias(request.problem.hasBias);
  tilingApi.SetTraverse(toMatmulTraverse(request.hints.preferTraverse));
  tilingApi.SetFixSplit(*mOr, *nOr, -1);
  tilingApi.SetBufferSpace(-1, -1, -1);

  optiling::TCubeTiling tilingData;
  (void)tilingApi.GetTiling(tilingData);

  MatmulTilingResult result;
  result.backendKind = "api";
  result.strategyName = "matmul-api";
  result.blockDim = static_cast<uint32_t>(tilingData.get_usedCoreNum());
  result.tilingData.resize(tilingData.GetDataSize());
  tilingData.SaveToBuffer(result.tilingData.data(), tilingData.GetDataSize());
  result.debugNote = "soc=" + socVersion + " traverse=" +
                     traverseToString(request.hints.preferTraverse) +
                     " bias=" + std::string(request.problem.hasBias ? "1" : "0") +
                     " bias_dtype=" +
                     (request.problem.hasBias
                          ? std::string(toString(resolveBiasDType(request)))
                          : std::string("none"));
  return result;
}

} // namespace

llvm::StringRef MatmulApiTilingBackend::name() const { return "matmul-api"; }

bool MatmulApiTilingBackend::supports(const MatmulTilingRequest &request) const {
  return supportsMatmulApiTilingRequest(request);
}

llvm::Expected<MatmulTilingResult>
MatmulApiTilingBackend::generate(const MatmulTilingRequest &request) const {
  return generateMatmulApiTilingImpl(request);
}

} // namespace mlir::runtime
