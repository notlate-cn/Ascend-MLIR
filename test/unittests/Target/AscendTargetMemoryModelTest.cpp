//===- AscendTargetMemoryModelTest.cpp -----------------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetMemoryModel.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using llvm::FailureOr;
using llvm::failed;
using llvm::succeeded;

static TargetProfile makeCompleteProfile() {
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
  return profile;
}

TEST(AscendTargetMemoryModelTest, BuildsLogicalPlacesAndCapacities) {
  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(makeCompleteProfile(), os);
  ASSERT_TRUE(succeeded(model));
  EXPECT_TRUE(model->supportsMemoryPlace(MemoryPlace::A1));
  EXPECT_FALSE(model->supportsMemoryPlace(MemoryPlace::GMFlat));
  ASSERT_TRUE(succeeded(model->getCapacity(MemoryPlace::A2)));
  EXPECT_EQ(model->getCapacity(MemoryPlace::A2)->staticCapacityBytes,
            64 * 1024);
}

TEST(AscendTargetMemoryModelTest, UsesDocumentedMemorySpaceEncodings) {
  EXPECT_EQ(static_cast<int>(MemoryPlace::GM), 0);
  EXPECT_EQ(static_cast<int>(MemoryPlace::A1), 1);
  EXPECT_EQ(static_cast<int>(MemoryPlace::A2), 2);
  EXPECT_EQ(static_cast<int>(MemoryPlace::B1), 3);
  EXPECT_EQ(static_cast<int>(MemoryPlace::B2), 4);
  EXPECT_EQ(static_cast<int>(MemoryPlace::CO1), 7);
  EXPECT_EQ(static_cast<int>(MemoryPlace::VECIN), 9);
  EXPECT_EQ(static_cast<int>(MemoryPlace::VECOUT), 10);
  EXPECT_EQ(static_cast<int>(MemoryPlace::VECCALC), 11);
  EXPECT_EQ(static_cast<int>(MemoryPlace::GMFlat), 22);
}

TEST(AscendTargetMemoryModelTest, ProvidesVisibilityAndAlignmentRules) {
  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(makeCompleteProfile(), os);
  ASSERT_TRUE(succeeded(model));
  EXPECT_TRUE(model->isPlaceVisibleTo(MemoryPlace::A2, ExecutionUnit::Cube));
  EXPECT_FALSE(
      model->isPlaceVisibleTo(MemoryPlace::A2, ExecutionUnit::Vector));
  ASSERT_TRUE(succeeded(model->getAlignment(MemoryPlace::GM)));
  EXPECT_EQ(model->getAlignment(MemoryPlace::GM)->addressAlignmentBytes, 32);
}

TEST(AscendTargetMemoryModelTest, BuildsRequiredDirectPathGraph) {
  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(makeCompleteProfile(), os);
  ASSERT_TRUE(succeeded(model));
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::GM, MemoryPlace::A1).size(),
            1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::GM, MemoryPlace::B1).size(),
            1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::A1, MemoryPlace::A2).size(),
            1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::B1, MemoryPlace::B2).size(),
            1u);
  EXPECT_EQ(
      model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::VECIN).size(), 1u);
  EXPECT_EQ(
      model->findDirectPaths(MemoryPlace::GM, MemoryPlace::VECIN).size(), 1u);
  EXPECT_EQ(
      model->findDirectPaths(MemoryPlace::VECOUT, MemoryPlace::GM).size(),
      1u);
  EXPECT_EQ(
      model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::GM).size(), 1u);
  EXPECT_EQ(
      model->findDirectPaths(MemoryPlace::GM, MemoryPlace::VECCALC).size(),
      0u);

  auto edge =
      model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::VECIN).front();
  ASSERT_TRUE(succeeded(model->getPathKind(edge)));
  EXPECT_EQ(*model->getPathKind(edge), PathKind::QueueTransfer);

  auto fixpipeEdge =
      model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::GM).front();
  ASSERT_TRUE(succeeded(model->getPathKind(fixpipeEdge)));
  EXPECT_EQ(*model->getPathKind(fixpipeEdge), PathKind::FixPipe);
}

TEST(AscendTargetMemoryModelTest, RejectsMissingRequiredPlace) {
  TargetProfile profile = makeCompleteProfile();
  profile.capacityBytes.erase(MemoryPlace::A1);

  std::string message;
  llvm::raw_string_ostream os(message);
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(profile, os);
  EXPECT_TRUE(failed(model));
  EXPECT_NE(message.find("missing required memory place A1"),
            std::string::npos);
}

TEST(AscendTargetMemoryModelTest, RejectsZeroCapacityRequiredPlace) {
  TargetProfile profile = makeCompleteProfile();
  profile.capacityBytes[MemoryPlace::VECIN] = 0;

  std::string message;
  llvm::raw_string_ostream os(message);
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(profile, os);
  EXPECT_TRUE(failed(model));
  EXPECT_NE(message.find("missing required memory place VECIN"),
            std::string::npos);
}
