#include "Runtime/Mix/MixTilingGenerator.h"
#include "Runtime/Mix/MatmulTilingDispatcher.h"
#include "Runtime/Mix/MatmulApiTilingBackend.h"
#include "Runtime/Mix/NativeMatmulTilingBackend.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>
#include <memory>
#include <optional>

namespace mlir::runtime {
namespace {

class MixTilingStrategy {
public:
  virtual ~MixTilingStrategy() = default;
  virtual llvm::StringRef name() const = 0;
  virtual bool matches(const MixTilingRequest &request) const = 0;
  virtual llvm::Expected<MixTilingResult>
  generate(const MixTilingRequest &request) const = 0;
};

static constexpr const char *kMatmul2DTilingStrategyName = "matmul-2d";
static constexpr const char *kBatchMatmulTilingStrategyName = "batch-matmul";

static bool isRank2(llvm::ArrayRef<int64_t> shape) { return shape.size() == 2; }
static bool isRank3(llvm::ArrayRef<int64_t> shape) { return shape.size() == 3; }

static bool hasPositiveShape(llvm::ArrayRef<int64_t> shape) {
  return llvm::all_of(shape, [](int64_t dim) { return dim > 0; });
}

static bool isValidBiasShape(llvm::ArrayRef<int64_t> biasShape,
                             int64_t expectedN) {
  // Accept rank-1 [N] or rank-2 [1, N] / [N, 1] (the latter when the source
  // bias was `tensor<1xNxf32>` and only the body unit dim was collapsed; the
  // matmul bias intrinsic treats the buffer as 1D of length N regardless).
  if (biasShape.size() == 1)
    return biasShape[0] == expectedN;
  if (biasShape.size() == 2)
    return (biasShape[0] == 1 && biasShape[1] == expectedN) ||
           (biasShape[0] == expectedN && biasShape[1] == 1);
  return false;
}

static llvm::Expected<MatmulLayout> parseMatmulLayout(llvm::StringRef layout,
                                                      llvm::StringRef attrName) {
  if (layout == "ND")
    return MatmulLayout::ND;
  if (layout == "NZ")
    return MatmulLayout::NZ;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported %s layout: %s",
                                 attrName.str().c_str(), layout.str().c_str());
}

static llvm::Expected<EpilogueKind>
parseEpilogueKind(llvm::StringRef epilogueKind) {
  if (epilogueKind.empty() || epilogueKind == "None")
    return EpilogueKind::None;
  if (epilogueKind == "BiasAdd")
    return EpilogueKind::BiasAdd;
  if (epilogueKind == "BiasAddRelu")
    return EpilogueKind::BiasAddRelu;
  if (epilogueKind == "BiasAddLeakyRelu")
    return EpilogueKind::BiasAddLeakyRelu;
  if (epilogueKind == "Relu")
    return EpilogueKind::Relu;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported matmul epilogue kind: %s",
                                 epilogueKind.str().c_str());
}

static llvm::Expected<MatmulTilingRequest>
buildExplicitMatmulTilingRequest(const MixTilingRequest &request,
                                 const MixAbiMatmulDesc &matmul) {
  const bool isBatchMatmul = matmul.opKind == "batch_matmul";
  if (matmul.opKind != "matmul" && !isBatchMatmul) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported mix matmul op kind: %s",
                                   matmul.opKind.c_str());
  }
  if (request.inputs.size() < 2 || request.inputs.size() > 3 ||
      request.outputs.size() != 1) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid explicit matmul mix request: expected 2 or 3 inputs and 1 output");
  }

  const auto &a = request.inputs[0].shape;
  const auto &b = request.inputs[1].shape;
  const auto &c = request.outputs[0].shape;
  int64_t derivedM = 0;
  int64_t derivedN = 0;
  int64_t derivedKFromA = 0;
  int64_t derivedKFromB = 0;
  if (isBatchMatmul) {
    if (!isRank3(a) || !isRank3(b) || !isRank3(c) || !hasPositiveShape(a) ||
        !hasPositiveShape(b) || !hasPositiveShape(c) ||
        matmul.batchShape.size() != 1 || a[0] != matmul.batchShape[0] ||
        b[0] != matmul.batchShape[0] || c[0] != matmul.batchShape[0]) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "invalid explicit batch matmul mix request: only positive rank-3 tensors with a single static batch dim are supported");
    }
    derivedM = matmul.transA ? a[2] : a[1];
    derivedKFromA = matmul.transA ? a[1] : a[2];
    derivedN = matmul.transB ? b[1] : b[2];
    derivedKFromB = matmul.transB ? b[2] : b[1];
    if (c[1] != derivedM || c[2] != derivedN) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "invalid explicit batch matmul mix request: shapes do not match annotated matmul semantics");
    }
  } else {
    if (!isRank2(a) || !isRank2(b) || !isRank2(c) || !hasPositiveShape(a) ||
        !hasPositiveShape(b) || !hasPositiveShape(c) ||
        !matmul.batchShape.empty()) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "invalid explicit matmul mix request: only positive rank-2 tensors without batch are supported");
    }
    derivedM = matmul.transA ? a[1] : a[0];
    derivedKFromA = matmul.transA ? a[0] : a[1];
    derivedN = matmul.transB ? b[0] : b[1];
    derivedKFromB = matmul.transB ? b[1] : b[0];
    if (c[0] != derivedM || c[1] != derivedN) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "invalid explicit matmul mix request: shapes do not match annotated matmul semantics");
    }
  }
  if (derivedKFromA != derivedKFromB) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid explicit %s mix request: shapes do not match annotated matmul semantics",
        isBatchMatmul ? "batch matmul" : "matmul");
  }
  if (matmul.hasBias) {
    if (request.inputs.size() < 3 ||
        !isValidBiasShape(request.inputs[2].shape, derivedN)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "invalid explicit matmul mix request: bias must be a 1D vector with length N");
    }
  }

  auto layoutAOr = parseMatmulLayout(matmul.layoutA, "A");
  if (!layoutAOr)
    return layoutAOr.takeError();
  auto layoutBOr = parseMatmulLayout(matmul.layoutB, "B");
  if (!layoutBOr)
    return layoutBOr.takeError();
  auto layoutCOr = parseMatmulLayout(matmul.layoutC, "C");
  if (!layoutCOr)
    return layoutCOr.takeError();
  auto epilogueOr = parseEpilogueKind(matmul.epilogueKind);
  if (!epilogueOr)
    return epilogueOr.takeError();

  MatmulTilingRequest matmulRequest;
  matmulRequest.kernelName = request.kernelName;
  matmulRequest.problem.M = derivedM;
  matmulRequest.problem.N = derivedN;
  matmulRequest.problem.K = derivedKFromA;
  matmulRequest.problem.batchShape = matmul.batchShape;
  matmulRequest.problem.dtypeA = request.inputs[0].dtype;
  matmulRequest.problem.dtypeB = request.inputs[1].dtype;
  matmulRequest.problem.dtypeC = request.outputs[0].dtype;
  matmulRequest.problem.layoutA = *layoutAOr;
  matmulRequest.problem.layoutB = *layoutBOr;
  matmulRequest.problem.layoutC = *layoutCOr;
  matmulRequest.problem.transA = matmul.transA;
  matmulRequest.problem.transB = matmul.transB;
  matmulRequest.problem.hasBias = matmul.hasBias;
  if (matmul.hasBias)
    matmulRequest.problem.biasDType = request.inputs[2].dtype;
  matmulRequest.fusion.epilogue = *epilogueOr;
  matmulRequest.hints.socVersion = request.socVersion;
  return matmulRequest;
}

static llvm::Expected<MatmulTilingRequest>
buildMatmulTilingRequest(const MixTilingRequest &request) {
  if (request.matmul)
    return buildExplicitMatmulTilingRequest(request, *request.matmul);
  if (request.inputs.size() < 2 || request.inputs.size() > 3 ||
      request.outputs.size() != 1) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid matmul mix request: expected 2 or 3 inputs and 1 output");
  }
  const auto &a = request.inputs[0].shape;
  const auto &b = request.inputs[1].shape;
  const auto &c = request.outputs[0].shape;
  if (!isRank2(a) || !isRank2(b) || !isRank2(c) || !hasPositiveShape(a) ||
      !hasPositiveShape(b) || !hasPositiveShape(c)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid matmul mix request: only positive rank-2 tensors are supported");
  }
  if (request.inputs.size() > 2 &&
      !isValidBiasShape(request.inputs[2].shape, c[1])) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid matmul mix request: bias must be a 1D vector with length N");
  }

  MatmulTilingRequest matmulRequest;
  matmulRequest.kernelName = request.kernelName;
  matmulRequest.problem.M = c[0];
  matmulRequest.problem.N = c[1];
  matmulRequest.problem.K = a[1];
  matmulRequest.problem.dtypeA = request.inputs[0].dtype;
  matmulRequest.problem.dtypeB = request.inputs[1].dtype;
  matmulRequest.problem.dtypeC = request.outputs[0].dtype;
  if (request.inputs.size() > 2) {
    matmulRequest.problem.hasBias = true;
    matmulRequest.problem.biasDType = request.inputs[2].dtype;
  }
  matmulRequest.problem.transA = false;
  matmulRequest.problem.transB = false;
  matmulRequest.problem.layoutA = MatmulLayout::ND;
  matmulRequest.problem.layoutB = MatmulLayout::ND;
  matmulRequest.problem.layoutC = MatmulLayout::ND;
  matmulRequest.hints.socVersion = request.socVersion;
  return matmulRequest;
}

class Matmul2DTilingStrategy final : public MixTilingStrategy {
public:
  llvm::StringRef name() const override { return "matmul-2d"; }

  bool matches(const MixTilingRequest &request) const override {
    if (request.matmul) {
      auto explicitMatmulRequest = buildExplicitMatmulTilingRequest(
          request, *request.matmul);
      return static_cast<bool>(explicitMatmulRequest);
    }
    if (request.inputs.size() < 2 || request.inputs.size() > 3 ||
        request.outputs.size() != 1)
      return false;
    const auto &a = request.inputs[0].shape;
    const auto &b = request.inputs[1].shape;
    const auto &c = request.outputs[0].shape;
    if (!isRank2(a) || !isRank2(b) || !isRank2(c))
      return false;
    if (!hasPositiveShape(a) || !hasPositiveShape(b) || !hasPositiveShape(c))
      return false;
    return a[1] == b[0] && c[0] == a[0] && c[1] == b[1];
  }

  llvm::Expected<MixTilingResult>
  generate(const MixTilingRequest &request) const override {
    auto matmulRequestOr = buildMatmulTilingRequest(request);
    if (!matmulRequestOr)
      return matmulRequestOr.takeError();

    NativeMatmulTilingBackend nativeBackend;
    MatmulApiTilingBackend apiBackend;
    auto resultOr = dispatchMatmulTiling(*matmulRequestOr, nativeBackend,
                                         apiBackend);
    if (!resultOr)
      return resultOr.takeError();

    MixTilingResult result;
    result.backendKind = resultOr->backendKind;
    result.strategyName = kMatmul2DTilingStrategyName;
    result.blockDim = resultOr->blockDim;
    result.tilingData = resultOr->tilingData;
    result.debugNote = resultOr->debugNote;
    return result;
  }
};

class BatchMatmulTilingStrategy final : public MixTilingStrategy {
public:
  llvm::StringRef name() const override { return kBatchMatmulTilingStrategyName; }

  bool matches(const MixTilingRequest &request) const override {
    return request.matmul && request.matmul->opKind == "batch_matmul";
  }

  llvm::Expected<MixTilingResult>
  generate(const MixTilingRequest &request) const override {
    auto matmulRequestOr = buildMatmulTilingRequest(request);
    if (!matmulRequestOr)
      return matmulRequestOr.takeError();

    NativeMatmulTilingBackend nativeBackend;
    MatmulApiTilingBackend apiBackend;
    auto resultOr = dispatchMatmulTiling(*matmulRequestOr, nativeBackend,
                                         apiBackend);
    if (!resultOr)
      return resultOr.takeError();

    MixTilingResult result;
    result.backendKind = resultOr->backendKind;
    result.strategyName = kBatchMatmulTilingStrategyName;
    result.blockDim = resultOr->blockDim;
    result.tilingData = resultOr->tilingData;
    result.debugNote = resultOr->debugNote;
    return result;
  }
};

static llvm::ArrayRef<std::unique_ptr<MixTilingStrategy>> getStrategies() {
  static std::vector<std::unique_ptr<MixTilingStrategy>> strategies = [] {
    std::vector<std::unique_ptr<MixTilingStrategy>> out;
    out.push_back(std::make_unique<BatchMatmulTilingStrategy>());
    out.push_back(std::make_unique<Matmul2DTilingStrategy>());
    return out;
  }();
  return strategies;
}

static llvm::Error ensureParentDirectory(llvm::StringRef path) {
  llvm::SmallString<256> parent = llvm::sys::path::parent_path(path);
  if (parent.empty())
    return llvm::Error::success();
  if (auto ec = llvm::sys::fs::create_directories(parent))
    return llvm::createStringError(ec, "cannot create parent directory for %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

static llvm::Error writeBinaryFile(llvm::StringRef path,
                                   llvm::ArrayRef<uint8_t> data) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open output file: %s",
                                   path.str().c_str());
  os.write(reinterpret_cast<const char *>(data.data()),
           static_cast<std::streamsize>(data.size()));
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write output file: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

static llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open output file: %s",
                                   path.str().c_str());
  os << content.str();
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write output file: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

} // namespace

llvm::StringRef getDefaultMixTilingBackendName() { return "in-process"; }

llvm::Expected<MixTilingResult>
generateMixTilingInProcess(const MixTilingRequest &request) {
  for (const auto &strategy : getStrategies()) {
    if (strategy->matches(request))
      return strategy->generate(request);
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "unsupported mix tiling request: no registered strategy matches kernel %s",
      request.kernelName.c_str());
}

llvm::Error writeMixTilingArtifacts(const MixTilingResult &result,
                                    llvm::StringRef tilingOutputPath,
                                    llvm::StringRef launchInfoOutputPath) {
  if (auto err = ensureParentDirectory(tilingOutputPath))
    return err;
  if (auto err = ensureParentDirectory(launchInfoOutputPath))
    return err;
  if (auto err = writeBinaryFile(tilingOutputPath, result.tilingData))
    return err;

  std::string launchInfo =
      ("block_dim=" + std::to_string(result.blockDim) + "\n");
  return writeTextFile(launchInfoOutputPath, launchInfo);
}

} // namespace mlir::runtime
