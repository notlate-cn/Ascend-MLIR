#include "Runtime/Mix/MixTilingGenerator.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

namespace {

llvm::cl::opt<std::string> KernelName("name", llvm::cl::Required);
llvm::cl::opt<std::string> SocVersion("soc", llvm::cl::init("Ascend910B1"));
llvm::cl::opt<std::string> AShape("a-shape", llvm::cl::Required);
llvm::cl::opt<std::string> ADType("a-dtype", llvm::cl::Required);
llvm::cl::opt<std::string> BShape("b-shape", llvm::cl::Required);
llvm::cl::opt<std::string> BDType("b-dtype", llvm::cl::Required);
llvm::cl::opt<std::string> CShape("c-shape", llvm::cl::Required);
llvm::cl::opt<std::string> CDType("c-dtype", llvm::cl::Required);
llvm::cl::opt<std::string> BiasDType("bias-dtype", llvm::cl::init(""));
llvm::cl::opt<std::string> TilingOut("tiling-out", llvm::cl::Required);
llvm::cl::opt<std::string> LaunchInfoOut("launch-info-out", llvm::cl::Required);

llvm::Expected<std::vector<int64_t>> parseShape(llvm::StringRef raw) {
  llvm::SmallVector<llvm::StringRef> parts;
  raw.split(parts, ',', -1, false);
  std::vector<int64_t> shape;
  shape.reserve(parts.size());
  for (llvm::StringRef part : parts) {
    int64_t value = 0;
    if (part.getAsInteger(10, value))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid shape component: %s",
                                     part.str().c_str());
    shape.push_back(value);
  }
  return shape;
}

llvm::Expected<mlir::runtime::DType> parseRuntimeDType(llvm::StringRef raw) {
  if (raw == "f16")
    return mlir::runtime::DType::F16;
  if (raw == "bf16")
    return mlir::runtime::DType::BF16;
  if (raw == "f32")
    return mlir::runtime::DType::F32;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported helper dtype: %s",
                                 raw.str().c_str());
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Emit mix matmul tiling.bin and launch_info.txt\n");

  auto aShapeOr = parseShape(AShape);
  if (!aShapeOr) {
    llvm::errs() << llvm::toString(aShapeOr.takeError()) << "\n";
    return 2;
  }
  auto bShapeOr = parseShape(BShape);
  if (!bShapeOr) {
    llvm::errs() << llvm::toString(bShapeOr.takeError()) << "\n";
    return 2;
  }
  auto cShapeOr = parseShape(CShape);
  if (!cShapeOr) {
    llvm::errs() << llvm::toString(cShapeOr.takeError()) << "\n";
    return 2;
  }
  if (aShapeOr->size() != 2 || bShapeOr->size() != 2 || cShapeOr->size() != 2) {
    llvm::errs() << "mix-tiling-helper currently expects 2D matmul shapes\n";
    return 2;
  }

  auto aDTypeOr = parseRuntimeDType(ADType);
  if (!aDTypeOr) {
    llvm::errs() << llvm::toString(aDTypeOr.takeError()) << "\n";
    return 2;
  }
  auto bDTypeOr = parseRuntimeDType(BDType);
  if (!bDTypeOr) {
    llvm::errs() << llvm::toString(bDTypeOr.takeError()) << "\n";
    return 2;
  }
  auto cDTypeOr = parseRuntimeDType(CDType);
  if (!cDTypeOr) {
    llvm::errs() << llvm::toString(cDTypeOr.takeError()) << "\n";
    return 2;
  }

  std::optional<mlir::runtime::DType> biasDTypeOr;
  if (!BiasDType.empty()) {
    auto parsedBiasOr = parseRuntimeDType(BiasDType);
    if (!parsedBiasOr) {
      llvm::errs() << llvm::toString(parsedBiasOr.takeError()) << "\n";
      return 2;
    }
    biasDTypeOr = *parsedBiasOr;
  }

  mlir::runtime::MixTilingRequest request;
  request.kernelName = KernelName;
  request.socVersion = SocVersion;
  request.inputs.push_back({*aDTypeOr, *aShapeOr});
  request.inputs.push_back({*bDTypeOr, *bShapeOr});
  if (biasDTypeOr)
    request.inputs.push_back({*biasDTypeOr, {}});
  request.outputs.push_back({*cDTypeOr, *cShapeOr});

  auto tilingOr = mlir::runtime::generateMixTilingInProcess(request);
  if (!tilingOr) {
    llvm::errs() << llvm::toString(tilingOr.takeError()) << "\n";
    return 2;
  }
  if (auto err =
          mlir::runtime::writeMixTilingArtifacts(*tilingOr, TilingOut,
                                                 LaunchInfoOut)) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    return 2;
  }

  llvm::outs() << "kernel_name=" << KernelName << "\n";
  llvm::outs() << "tiling_out=" << TilingOut << "\n";
  llvm::outs() << "launch_info_out=" << LaunchInfoOut << "\n";
  return 0;
}
