#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace matmul_tiling;

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

llvm::Expected<DataType> parseAclDataType(llvm::StringRef raw) {
  if (raw == "f16")
    return DataType::DT_FLOAT16;
  if (raw == "bf16")
    return DataType::DT_BF16;
  if (raw == "f32")
    return DataType::DT_FLOAT;
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported helper dtype: %s",
                                 raw.str().c_str());
}

llvm::Error writeBinaryFile(llvm::StringRef path, const void *data, size_t size) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot open output file: %s",
                                   path.str().c_str());
  os.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write output file: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content) {
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

llvm::Error ensureParentDirectory(llvm::StringRef path) {
  llvm::SmallString<256> parent =
      llvm::sys::path::parent_path(llvm::StringRef(path));
  if (parent.empty())
    return llvm::Error::success();
  if (auto ec = llvm::sys::fs::create_directories(parent))
    return llvm::createStringError(ec, "cannot create parent directory for %s",
                                   path.str().c_str());
  return llvm::Error::success();
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

  auto aDTypeOr = parseAclDataType(ADType);
  if (!aDTypeOr) {
    llvm::errs() << llvm::toString(aDTypeOr.takeError()) << "\n";
    return 2;
  }
  auto bDTypeOr = parseAclDataType(BDType);
  if (!bDTypeOr) {
    llvm::errs() << llvm::toString(bDTypeOr.takeError()) << "\n";
    return 2;
  }
  auto cDTypeOr = parseAclDataType(CDType);
  if (!cDTypeOr) {
    llvm::errs() << llvm::toString(cDTypeOr.takeError()) << "\n";
    return 2;
  }

  std::optional<DataType> biasDTypeOr;
  if (!BiasDType.empty()) {
    auto parsedBiasOr = parseAclDataType(BiasDType);
    if (!parsedBiasOr) {
      llvm::errs() << llvm::toString(parsedBiasOr.takeError()) << "\n";
      return 2;
    }
    biasDTypeOr = *parsedBiasOr;
  }

  auto *ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SocVersion.c_str());
  if (!ascendcPlatform) {
    llvm::errs() << "cannot initialize AscendC platform for soc "
                 << SocVersion << "\n";
    return 2;
  }

  MatmulApiTiling tilingApi(*ascendcPlatform);
  tilingApi.SetAType(TPosition::GM, CubeFormat::ND, *aDTypeOr, false);
  tilingApi.SetBType(TPosition::GM, CubeFormat::ND, *bDTypeOr, false);
  tilingApi.SetCType(TPosition::GM, CubeFormat::ND, *cDTypeOr);
  if (biasDTypeOr)
    tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, *biasDTypeOr);
  tilingApi.SetOrgShape(static_cast<int>((*cShapeOr)[0]),
                        static_cast<int>((*cShapeOr)[1]),
                        static_cast<int>((*aShapeOr)[1]));
  tilingApi.SetShape(static_cast<int>((*cShapeOr)[0]),
                     static_cast<int>((*cShapeOr)[1]),
                     static_cast<int>((*aShapeOr)[1]));
  tilingApi.SetBias(biasDTypeOr.has_value());
  tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
  tilingApi.SetFixSplit(static_cast<int>((*cShapeOr)[0]),
                        static_cast<int>((*cShapeOr)[1]), -1);
  tilingApi.SetBufferSpace(-1, -1, -1);

  optiling::TCubeTiling tilingData;
  (void)tilingApi.GetTiling(tilingData);

  if (auto err = ensureParentDirectory(TilingOut)) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    return 2;
  }
  if (auto err = ensureParentDirectory(LaunchInfoOut)) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    return 2;
  }

  std::vector<uint8_t> tilingBuffer(sizeof(optiling::TCubeTiling), 0);
  tilingData.SaveToBuffer(tilingBuffer.data(), tilingData.GetDataSize());
  if (auto err =
          writeBinaryFile(TilingOut, tilingBuffer.data(), tilingBuffer.size())) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    return 2;
  }

  std::string launchInfo =
      ("block_dim=" + std::to_string(tilingData.get_usedCoreNum()) + "\n");
  if (auto err = writeTextFile(LaunchInfoOut, launchInfo)) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    return 2;
  }

  llvm::outs() << "kernel_name=" << KernelName << "\n";
  llvm::outs() << "tiling_out=" << TilingOut << "\n";
  llvm::outs() << "launch_info_out=" << LaunchInfoOut << "\n";
  return 0;
}
