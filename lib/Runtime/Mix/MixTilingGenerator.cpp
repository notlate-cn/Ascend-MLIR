#include "Runtime/Mix/MixTilingGenerator.h"
#include "Runtime/Mix/MatmulApiTilingBackend.h"

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

static bool isRank2(llvm::ArrayRef<int64_t> shape) { return shape.size() == 2; }

static bool hasPositiveShape(llvm::ArrayRef<int64_t> shape) {
  return llvm::all_of(shape, [](int64_t dim) { return dim > 0; });
}

class Matmul2DTilingStrategy final : public MixTilingStrategy {
public:
  llvm::StringRef name() const override { return "matmul-2d"; }

  bool matches(const MixTilingRequest &request) const override {
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
    auto matmulRequest = buildMatmulApiTilingRequest(request);
    auto resultOr = generateMatmulApiTiling(matmulRequest);
    if (!resultOr)
      return resultOr.takeError();

    MixTilingResult result;
    result.blockDim = resultOr->blockDim;
    result.tilingData = resultOr->tilingData;
    result.strategyName = kMatmul2DTilingStrategyName;
    return result;
  }
};

static llvm::ArrayRef<std::unique_ptr<MixTilingStrategy>> getStrategies() {
  static std::vector<std::unique_ptr<MixTilingStrategy>> strategies = [] {
    std::vector<std::unique_ptr<MixTilingStrategy>> out;
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
