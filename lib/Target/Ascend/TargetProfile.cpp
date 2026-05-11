//===- TargetProfile.cpp - Ascend target profile --------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetProfile.h"

#include "Target/Ascend/CannTargetProfileLoader.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>

#define GEN_PASS_DECL_ASCENDPRINTTARGETPROFILEPASS
#define GEN_PASS_DEF_ASCENDPRINTTARGETPROFILEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::ascend {

StringRef stringifyMemoryPlace(MemoryPlace place) {
  switch (place) {
  case MemoryPlace::GM:
    return "GM";
  case MemoryPlace::A1:
    return "A1";
  case MemoryPlace::A2:
    return "A2";
  case MemoryPlace::B1:
    return "B1";
  case MemoryPlace::B2:
    return "B2";
  case MemoryPlace::CO1:
    return "CO1";
  case MemoryPlace::VECIN:
    return "VECIN";
  case MemoryPlace::VECOUT:
    return "VECOUT";
  case MemoryPlace::VECCALC:
    return "VECCALC";
  case MemoryPlace::GMFlat:
    return "GM_FLAT";
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
    FailureOr<ascend::TargetMemoryModel> memoryModel =
        ascend::TargetMemoryModelBuilder().build(profile, llvm::errs());
    if (failed(memoryModel)) {
      signalPassFailure();
      return;
    }

    llvm::errs() << "TargetProfile\n";
    llvm::errs() << "  soc = \"" << profile.identity.socVersion << "\"\n";
    if (!profile.identity.shortSocVersion.empty())
      llvm::errs() << "  short_soc = \"" << profile.identity.shortSocVersion
                   << "\"\n";
    if (!profile.identity.npuArch.empty())
      llvm::errs() << "  npu_arch = \"" << profile.identity.npuArch << "\"\n";

    for (ascend::MemoryPlace place : memoryModel->getMemoryPlaces()) {
      FailureOr<ascend::CapacityRule> capacity = memoryModel->getCapacity(place);
      if (failed(capacity))
        continue;
      llvm::errs() << "  memory_place = \""
                   << ascend::stringifyMemoryPlace(place) << "\"";
      if (capacity->staticCapacityBytes > 0)
        llvm::errs() << " capacity_bytes = "
                     << capacity->staticCapacityBytes;
      llvm::errs() << "\n";
    }

    SmallVector<StringRef> intrinsicNames;
    intrinsicNames.reserve(profile.intrinsics.size());
    for (const ascend::TargetIntrinsicInfo &intrinsic : profile.intrinsics)
      intrinsicNames.push_back(intrinsic.name);
    llvm::sort(intrinsicNames);

    for (StringRef intrinsicName : intrinsicNames)
      llvm::errs() << "  intrinsic = \"" << intrinsicName << "\"\n";

    SmallVector<const ascend::TargetMemoryRateInfo *> memoryRates;
    memoryRates.reserve(profile.memoryRates.size());
    for (const ascend::TargetMemoryRateInfo &rate : profile.memoryRates)
      memoryRates.push_back(&rate);
    llvm::sort(memoryRates, [](const ascend::TargetMemoryRateInfo *lhs,
                               const ascend::TargetMemoryRateInfo *rhs) {
      return std::tie(lhs->section, lhs->name) <
             std::tie(rhs->section, rhs->name);
    });

    for (const ascend::TargetMemoryRateInfo *rate : memoryRates) {
      llvm::errs() << "  memory_rate = \"" << rate->section << "."
                   << rate->name << "\" bytes_per_cycle = "
                   << rate->bytesPerCycle << "\n";
    }
  }
};

} // namespace

namespace mlir::afir {

std::unique_ptr<Pass> createAscendPrintTargetProfilePass() {
  return std::make_unique<AscendPrintTargetProfilePass>();
}

} // namespace mlir::afir
