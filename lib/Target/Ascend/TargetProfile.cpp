//===- TargetProfile.cpp - Ascend target profile --------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetProfile.h"

#include "Target/Ascend/CannTargetProfileLoader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DECL_ASCENDPRINTTARGETPROFILEPASS
#define GEN_PASS_DEF_ASCENDPRINTTARGETPROFILEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::ascend {

StringRef stringifyMemoryPlace(MemoryPlace place) {
  switch (place) {
  case MemoryPlace::GM:
    return "GM";
  case MemoryPlace::L2:
    return "L2";
  case MemoryPlace::L1:
    return "L1";
  case MemoryPlace::L0A:
    return "L0A";
  case MemoryPlace::L0B:
    return "L0B";
  case MemoryPlace::L0C:
    return "L0C";
  case MemoryPlace::UB:
    return "UB";
  }
  llvm_unreachable("unknown memory place");
}

LogicalResult verifyTargetProfile(const TargetProfile &profile,
                                  raw_ostream &os) {
  if (profile.identity.socVersion.empty()) {
    os << "TargetProfile verification failed: missing SoC version\n";
    return failure();
  }
  if (profile.hardware.aiCoreCount <= 0) {
    os << "TargetProfile verification failed: ai_core_cnt must be nonzero\n";
    return failure();
  }
  if (profile.hardware.l1SizeBytes <= 0) {
    os << "TargetProfile verification failed: l1_size must be nonzero\n";
    return failure();
  }
  if (profile.hardware.ubSizeBytes <= 0) {
    os << "TargetProfile verification failed: ub_size must be nonzero\n";
    return failure();
  }
  return success();
}

} // namespace mlir::ascend

namespace {

constexpr ascend::MemoryPlace orderedMemoryPlaces[] = {
    ascend::MemoryPlace::GM,  ascend::MemoryPlace::L2,  ascend::MemoryPlace::L1,
    ascend::MemoryPlace::L0A, ascend::MemoryPlace::L0B, ascend::MemoryPlace::L0C,
    ascend::MemoryPlace::UB};

struct AscendPrintTargetProfilePass
    : public ::impl::AscendPrintTargetProfilePassBase<
          AscendPrintTargetProfilePass> {
  using AscendPrintTargetProfilePassBase::AscendPrintTargetProfilePassBase;

  void runOnOperation() override {
    FailureOr<ascend::TargetProfile> loaded =
        ascend::CannTargetProfileLoader::load(cannRoot, soc);
    if (failed(loaded)) {
      signalPassFailure();
      return;
    }

    const ascend::TargetProfile &profile = *loaded;
    llvm::errs() << "TargetProfile\n";
    llvm::errs() << "  soc = \"" << profile.identity.socVersion << "\"\n";
    if (!profile.identity.shortSocVersion.empty())
      llvm::errs() << "  short_soc = \"" << profile.identity.shortSocVersion
                   << "\"\n";
    if (!profile.identity.npuArch.empty())
      llvm::errs() << "  npu_arch = \"" << profile.identity.npuArch << "\"\n";

    for (ascend::MemoryPlace place : orderedMemoryPlaces) {
      auto it = profile.capacityBytes.find(place);
      if (it == profile.capacityBytes.end())
        continue;
      llvm::errs() << "  memory_place = \""
                   << ascend::stringifyMemoryPlace(place) << "\"";
      if (it->second > 0)
        llvm::errs() << " capacity_bytes = " << it->second;
      llvm::errs() << "\n";
    }

    SmallVector<StringRef> intrinsicNames;
    intrinsicNames.reserve(profile.intrinsics.size());
    for (const ascend::TargetIntrinsicInfo &intrinsic : profile.intrinsics)
      intrinsicNames.push_back(intrinsic.name);
    llvm::sort(intrinsicNames);

    for (StringRef intrinsicName : intrinsicNames)
      llvm::errs() << "  intrinsic = \"" << intrinsicName << "\"\n";
  }
};

} // namespace

namespace mlir::afir {

std::unique_ptr<Pass> createAscendPrintTargetProfilePass() {
  return std::make_unique<AscendPrintTargetProfilePass>();
}

} // namespace mlir::afir
