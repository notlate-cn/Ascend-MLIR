//===- AscendTargetCostModelTest.cpp -------------------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/CannTargetProfileLoader.h"
#include "Target/Ascend/TargetCostModel.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using llvm::FailureOr;
using llvm::failed;
using llvm::succeeded;

namespace {

struct TempDir {
  llvm::SmallString<256> path;

  ~TempDir() {
    if (!path.empty())
      (void)llvm::sys::fs::remove_directories(path);
  }
};

static TargetProfile makeProfileWithRates() {
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

static FailureOr<TargetCostModel>
buildCostModel(const TargetProfile &profile,
               const TargetMemoryModel &memoryModel,
               std::string *message = nullptr) {
  if (!message) {
    llvm::raw_null_ostream os;
    return TargetCostModelBuilder().build(profile, memoryModel, os);
  }

  llvm::raw_string_ostream os(*message);
  return TargetCostModelBuilder().build(profile, memoryModel, os);
}

static PathEdge onlyDirectPath(const TargetMemoryModel &model, MemoryPlace src,
                               MemoryPlace dst) {
  llvm::SmallVector<PathEdge> paths = model.findDirectPaths(src, dst);
  if (paths.empty()) {
    ADD_FAILURE() << "missing direct path";
    return {src, dst, ~0u};
  }
  EXPECT_EQ(paths.size(), 1u);
  return paths.front();
}

static const TargetMemoryRateInfo *findRate(const TargetProfile &profile,
                                            llvm::StringRef section,
                                            llvm::StringRef name) {
  for (const TargetMemoryRateInfo &rate : profile.memoryRates)
    if (rate.section == section && rate.name == name)
      return &rate;
  return nullptr;
}

static void writeSyntheticCannProfile(llvm::StringRef root) {
  llvm::SmallString<256> configDir(root);
  llvm::sys::path::append(configDir, "aarch64-linux", "data",
                          "platform_config");
  ASSERT_FALSE(llvm::sys::fs::create_directories(configDir));

  llvm::SmallString<256> iniPath(configDir);
  llvm::sys::path::append(iniPath, "SyntheticSoC.ini");

  std::error_code ec;
  llvm::raw_fd_ostream os(iniPath, ec);
  ASSERT_FALSE(ec) << ec.message();

  os << R"ini(
[version]
SoC_version=SyntheticSoC
Short_SoC_version=SYN
NpuArch=SYNTH

[SoCInfo]
ai_core_cnt=1
memory_size=1048576
support_bf16=true

[AICoreSpec]
l1_size=524288
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
ub_size=196608
support_fixpipe=true

[AICoreMemoryRates]
ddr_rate=32
ddr_read_rate=32
l1_to_l0_a_rate=512
Intrinsic_data_move_out2l1=64

[VectorCoreMemoryRates]
ub_to_ddr_rate=96
)ini";
}

} // namespace

TEST(AscendTargetCostModelTest, LooksUpExactAndPreferredRates) {
  TargetProfile profile = makeProfileWithRates();
  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetCostModel> costModel = buildCostModel(profile, *memoryModel);
  ASSERT_TRUE(succeeded(costModel));

  ASSERT_TRUE(
      succeeded(costModel->getMemoryRate("AICoreMemoryRates", "ddr_rate")));
  EXPECT_EQ(*costModel->getMemoryRate("AICoreMemoryRates", "ddr_rate"), 32);
  EXPECT_EQ(*costModel->getMemoryRate("VectorCoreMemoryRates",
                                      "ub_to_ddr_rate"),
            96);
  EXPECT_EQ(*costModel->getPreferredMemoryRate("ub_to_ddr_rate"), 64);
  EXPECT_TRUE(
      failed(costModel->getMemoryRate("AICoreMemoryRates", "missing_rate")));
}

TEST(AscendTargetCostModelTest, BuildsPathCostsForDirectEdges) {
  TargetProfile profile = makeProfileWithRates();
  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetCostModel> costModel = buildCostModel(profile, *memoryModel);
  ASSERT_TRUE(succeeded(costModel));

  PathEdge gmToA1 =
      onlyDirectPath(*memoryModel, MemoryPlace::GM, MemoryPlace::A1);
  FailureOr<PathCost> gmToA1Cost = costModel->getPathCost(gmToA1);
  ASSERT_TRUE(succeeded(gmToA1Cost));
  EXPECT_EQ(gmToA1Cost->rateSection, "AICoreMemoryRates");
  EXPECT_EQ(gmToA1Cost->rateName, "ddr_read_rate");
  EXPECT_EQ(gmToA1Cost->bytesPerCycle, 32);
  EXPECT_EQ(gmToA1Cost->startupCycles, 2);
  EXPECT_FALSE(gmToA1Cost->overlapsCompute);
  EXPECT_EQ(*costModel->estimateTransferCycles(gmToA1, 64), 4);

  PathEdge a1ToA2 =
      onlyDirectPath(*memoryModel, MemoryPlace::A1, MemoryPlace::A2);
  FailureOr<PathCost> a1ToA2Cost = costModel->getPathCost(a1ToA2);
  ASSERT_TRUE(succeeded(a1ToA2Cost));
  EXPECT_EQ(a1ToA2Cost->rateSection, "AICoreMemoryRates");
  EXPECT_EQ(a1ToA2Cost->rateName, "l1_to_l0_a_rate");
  EXPECT_EQ(a1ToA2Cost->bytesPerCycle, 512);
  EXPECT_EQ(*costModel->estimateTransferCycles(a1ToA2, 1024), 3);

  PathEdge queue =
      onlyDirectPath(*memoryModel, MemoryPlace::CO1, MemoryPlace::VECIN);
  FailureOr<PathCost> queueCost = costModel->getPathCost(queue);
  ASSERT_TRUE(succeeded(queueCost));
  EXPECT_EQ(queueCost->rateName, "l0_c_to_ub_rate");
  EXPECT_TRUE(queueCost->overlapsCompute);

  PathEdge gmToVecin =
      onlyDirectPath(*memoryModel, MemoryPlace::GM, MemoryPlace::VECIN);
  FailureOr<PathCost> gmToVecinCost = costModel->getPathCost(gmToVecin);
  ASSERT_TRUE(succeeded(gmToVecinCost));
  EXPECT_EQ(gmToVecinCost->rateSection, "VectorCoreMemoryRates");
  EXPECT_EQ(gmToVecinCost->rateName, "ddr_read_rate");
  EXPECT_EQ(gmToVecinCost->bytesPerCycle, 48);

  PathEdge vecoutToGm =
      onlyDirectPath(*memoryModel, MemoryPlace::VECOUT, MemoryPlace::GM);
  FailureOr<PathCost> vecoutToGmCost = costModel->getPathCost(vecoutToGm);
  ASSERT_TRUE(succeeded(vecoutToGmCost));
  EXPECT_EQ(vecoutToGmCost->rateSection, "VectorCoreMemoryRates");
  EXPECT_EQ(vecoutToGmCost->rateName, "ub_to_ddr_rate");
  EXPECT_EQ(vecoutToGmCost->bytesPerCycle, 96);
  EXPECT_EQ(*costModel->estimateTransferCycles(vecoutToGm, 192), 3);
}

TEST(AscendTargetCostModelTest, RejectsMissingRequiredPathRate) {
  TargetProfile profile = makeProfileWithRates();
  llvm::erase_if(profile.memoryRates, [](const TargetMemoryRateInfo &rate) {
    return rate.name == "l1_to_l0_b_rate";
  });

  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));

  std::string message;
  FailureOr<TargetCostModel> costModel =
      buildCostModel(profile, *memoryModel, &message);
  EXPECT_TRUE(failed(costModel));
  EXPECT_NE(message.find("AICoreMemoryRates.l1_to_l0_b_rate"),
            std::string::npos);
  EXPECT_NE(message.find("VectorCoreMemoryRates.l1_to_l0_b_rate"),
            std::string::npos);
  EXPECT_NE(message.find("B1 -> B2"), std::string::npos);
}

TEST(AscendTargetCostModelTest, LoaderParsesMemoryRates) {
  TempDir temp;
  ASSERT_FALSE(
      llvm::sys::fs::createUniqueDirectory("ascend-cann-cost", temp.path));
  writeSyntheticCannProfile(temp.path);

  FailureOr<TargetProfile> profile =
      CannTargetProfileLoader::load(temp.path, "SyntheticSoC");
  ASSERT_TRUE(succeeded(profile));

  const TargetMemoryRateInfo *ddr =
      findRate(*profile, "AICoreMemoryRates", "ddr_rate");
  ASSERT_NE(ddr, nullptr);
  EXPECT_EQ(ddr->bytesPerCycle, 32);

  const TargetMemoryRateInfo *vectorRate =
      findRate(*profile, "VectorCoreMemoryRates", "ub_to_ddr_rate");
  ASSERT_NE(vectorRate, nullptr);
  EXPECT_EQ(vectorRate->bytesPerCycle, 96);

  EXPECT_EQ(
      findRate(*profile, "AICoreMemoryRates", "Intrinsic_data_move_out2l1"),
      nullptr);
}
