//===- AscendTargetMemoryModelTest.cpp -----------------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetMemoryModel.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using llvm::FailureOr;
using llvm::failed;
using llvm::SmallVector;
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
  profile.intrinsics.push_back({"Intrinsic_data_move_out2l1", {"f16", "f32"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_l12l0a", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_l12l0b", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_ub2out", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_fix_pipe_l0c2out", {"f16"}});
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

TEST(AscendTargetMemoryModelTest, FindsMultiHopRoutesAndPathConstraints) {
  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(makeCompleteProfile(), os);
  ASSERT_TRUE(succeeded(model));

  SmallVector<PathRoute> gmToA2Routes =
      model->findPaths(MemoryPlace::GM, MemoryPlace::A2);
  ASSERT_FALSE(gmToA2Routes.empty());
  ASSERT_EQ(gmToA2Routes.front().edges.size(), 2u);
  EXPECT_EQ(gmToA2Routes.front().edges[0].srcPlace, MemoryPlace::GM);
  EXPECT_EQ(gmToA2Routes.front().edges[0].dstPlace, MemoryPlace::A1);
  EXPECT_EQ(gmToA2Routes.front().edges[1].srcPlace, MemoryPlace::A1);
  EXPECT_EQ(gmToA2Routes.front().edges[1].dstPlace, MemoryPlace::A2);

  FailureOr<SmallVector<PathConstraint>> load2DConstraints =
      model->getPathConstraints(gmToA2Routes.front().edges[0]);
  ASSERT_TRUE(succeeded(load2DConstraints));
  ASSERT_FALSE(load2DConstraints->empty());
  EXPECT_TRUE(
      llvm::is_contained(load2DConstraints->front().dtypes, "f16"));
  EXPECT_EQ(load2DConstraints->front().minRank, 2);
  EXPECT_TRUE(load2DConstraints->front().requires2DLoad);
  EXPECT_FALSE(load2DConstraints->front().allowsTranspose);

  FailureOr<SmallVector<PathConstraint>> directConstraints =
      model->getPathConstraints(gmToA2Routes.front().edges[1]);
  ASSERT_TRUE(succeeded(directConstraints));
  ASSERT_FALSE(directConstraints->empty());
  EXPECT_TRUE(
      llvm::is_contained(directConstraints->front().dtypes, "f16"));
  EXPECT_EQ(directConstraints->front().minRank, 1);
  EXPECT_FALSE(directConstraints->front().requires2DLoad);

  SmallVector<PathRoute> directGmToVec =
      model->findPaths(MemoryPlace::GM, MemoryPlace::VECIN);
  ASSERT_FALSE(directGmToVec.empty());
  EXPECT_EQ(directGmToVec.front().edges.size(), 1u);
}

TEST(AscendTargetMemoryModelTest, AddsTransposePathVariantWhenAvailable) {
  TargetProfile profile = makeCompleteProfile();
  profile.intrinsics.push_back(
      {"Intrinsic_data_move_transpose_l12l0b", {"f16"}});

  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(profile, os);
  ASSERT_TRUE(succeeded(model));

  SmallVector<PathEdge> gmToB1 =
      model->findDirectPaths(MemoryPlace::GM, MemoryPlace::B1);
  ASSERT_EQ(gmToB1.size(), 2u);
  EXPECT_EQ(gmToB1[0].pathVariant, 0u);
  EXPECT_EQ(gmToB1[1].pathVariant, 1u);
  ASSERT_TRUE(succeeded(model->getPathKind(gmToB1[0])));
  ASSERT_TRUE(succeeded(model->getPathKind(gmToB1[1])));
  EXPECT_EQ(*model->getPathKind(gmToB1[0]), PathKind::Load2D);
  EXPECT_EQ(*model->getPathKind(gmToB1[1]), PathKind::Load2DTranspose);

  FailureOr<SmallVector<PathConstraint>> transposeConstraints =
      model->getPathConstraints(gmToB1[1]);
  ASSERT_TRUE(succeeded(transposeConstraints));
  ASSERT_FALSE(transposeConstraints->empty());
  EXPECT_TRUE(
      llvm::is_contained(transposeConstraints->front().dtypes, "f16"));
  EXPECT_EQ(transposeConstraints->front().minRank, 2);
  EXPECT_TRUE(transposeConstraints->front().requires2DLoad);
  EXPECT_TRUE(transposeConstraints->front().allowsTranspose);
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
