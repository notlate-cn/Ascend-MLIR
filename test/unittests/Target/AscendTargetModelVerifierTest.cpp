//===- AscendTargetModelVerifierTest.cpp ---------------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetCostModel.h"
#include "Target/Ascend/TargetIntrinsicModel.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetModelVerifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using llvm::FailureOr;
using llvm::failed;
using llvm::succeeded;

namespace {

static TargetProfile makeClosedProfile() {
  TargetProfile profile;
  profile.identity.socVersion = "TestSoC";
  profile.hardware.aiCoreCount = 1;
  profile.hardware.l1SizeBytes = 512 * 1024;
  profile.hardware.ubSizeBytes = 192 * 1024;
  profile.hardware.supportFixpipe = true;

  profile.capacityBytes[MemoryPlace::GM] = 1024 * 1024 * 1024;
  profile.capacityBytes[MemoryPlace::A1] = 512 * 1024;
  profile.capacityBytes[MemoryPlace::B1] = 512 * 1024;
  profile.capacityBytes[MemoryPlace::A2] = 64 * 1024;
  profile.capacityBytes[MemoryPlace::B2] = 64 * 1024;
  profile.capacityBytes[MemoryPlace::CO1] = 128 * 1024;
  profile.capacityBytes[MemoryPlace::VECIN] = 192 * 1024;
  profile.capacityBytes[MemoryPlace::VECOUT] = 192 * 1024;
  profile.capacityBytes[MemoryPlace::VECCALC] = 192 * 1024;

  profile.intrinsics.push_back({"Intrinsic_data_move_out2l1", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_out2l0a", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_l12l0a", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_l12l0b", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_ub2out", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_fix_pipe_l0c2out", {"f16"}});

  profile.memoryRates.push_back({"AICoreMemoryRates", "ddr_rate", 32});
  profile.memoryRates.push_back({"AICoreMemoryRates", "ddr_read_rate", 32});
  profile.memoryRates.push_back({"AICoreMemoryRates", "ddr_write_rate", 32});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "l1_to_l0_a_rate", 512});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "l1_to_l0_b_rate", 256});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "l0_c_to_ub_rate", 128});
  profile.memoryRates.push_back({"AICoreMemoryRates", "ub_to_ddr_rate", 64});
  profile.memoryRates.push_back({"VectorCoreMemoryRates", "ddr_read_rate", 48});
  profile.memoryRates.push_back(
      {"VectorCoreMemoryRates", "ub_to_ddr_rate", 96});
  return profile;
}

static FailureOr<TargetMemoryModel>
buildMemoryModel(const TargetProfile &profile) {
  llvm::raw_null_ostream os;
  return TargetMemoryModelBuilder().build(profile, os);
}

static FailureOr<TargetIntrinsicModel>
buildIntrinsicModel(const TargetProfile &profile) {
  return TargetIntrinsicModelBuilder().build(profile);
}

static FailureOr<TargetCostModel>
buildCostModel(const TargetProfile &profile,
               const TargetMemoryModel &memoryModel) {
  llvm::raw_null_ostream os;
  return TargetCostModelBuilder().build(profile, memoryModel, os);
}

static void expectDiagnosticContains(llvm::StringRef message,
                                     llvm::StringRef needle) {
  EXPECT_NE(message.find(needle), llvm::StringRef::npos)
      << "diagnostic did not contain '" << needle.str() << "': "
      << message.str();
}

} // namespace

TEST(AscendTargetModelVerifierTest, AcceptsClosedTargetModel) {
  TargetProfile profile = makeClosedProfile();
  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetIntrinsicModel> intrinsicModel = buildIntrinsicModel(profile);
  ASSERT_TRUE(succeeded(intrinsicModel));
  FailureOr<TargetCostModel> costModel = buildCostModel(profile, *memoryModel);
  ASSERT_TRUE(succeeded(costModel));

  std::string message;
  llvm::raw_string_ostream os(message);
  EXPECT_TRUE(mlir::succeeded(TargetModelVerifier().verify(
      profile, *memoryModel, *intrinsicModel, *costModel, os)));
  EXPECT_TRUE(message.empty());
}

TEST(AscendTargetModelVerifierTest, AcceptsClosedTransposePathVariant) {
  TargetProfile profile = makeClosedProfile();
  profile.intrinsics.push_back(
      {"Intrinsic_data_move_transpose_l12l0b", {"f16"}});

  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetIntrinsicModel> intrinsicModel = buildIntrinsicModel(profile);
  ASSERT_TRUE(succeeded(intrinsicModel));
  FailureOr<TargetCostModel> costModel = buildCostModel(profile, *memoryModel);
  ASSERT_TRUE(succeeded(costModel));

  std::string message;
  llvm::raw_string_ostream os(message);
  EXPECT_TRUE(mlir::succeeded(TargetModelVerifier().verify(
      profile, *memoryModel, *intrinsicModel, *costModel, os)));
  EXPECT_TRUE(message.empty());
}

TEST(AscendTargetModelVerifierTest, RejectsMissingMovementIntrinsic) {
  TargetProfile profile = makeClosedProfile();
  llvm::erase_if(profile.intrinsics, [](const TargetIntrinsicInfo &intrinsic) {
    return intrinsic.name == "Intrinsic_data_move_out2l1" ||
           intrinsic.name == "Intrinsic_data_move_out2l0a";
  });

  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetIntrinsicModel> intrinsicModel = buildIntrinsicModel(profile);
  ASSERT_TRUE(succeeded(intrinsicModel));
  FailureOr<TargetCostModel> costModel = buildCostModel(profile, *memoryModel);
  ASSERT_TRUE(succeeded(costModel));

  std::string message;
  llvm::raw_string_ostream os(message);
  EXPECT_TRUE(mlir::failed(TargetModelVerifier().verify(
      profile, *memoryModel, *intrinsicModel, *costModel, os)));
  os.flush();
  expectDiagnosticContains(message, "TargetPathIntrinsicMissing");
  expectDiagnosticContains(message, "Load2D");
}

TEST(AscendTargetModelVerifierTest, RejectsMissingPathCost) {
  TargetProfile profile = makeClosedProfile();
  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetIntrinsicModel> intrinsicModel = buildIntrinsicModel(profile);
  ASSERT_TRUE(succeeded(intrinsicModel));
  TargetCostModel emptyCostModel;

  std::string message;
  llvm::raw_string_ostream os(message);
  EXPECT_TRUE(mlir::failed(TargetModelVerifier().verify(
      profile, *memoryModel, *intrinsicModel, emptyCostModel, os)));
  os.flush();
  expectDiagnosticContains(message, "TargetPathCostMissing");
  expectDiagnosticContains(message, "GM -> A1");
}
