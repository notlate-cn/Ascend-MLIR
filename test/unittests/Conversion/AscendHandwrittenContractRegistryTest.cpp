//===- AscendHandwrittenContractRegistryTest.cpp --------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Pattern/HandwrittenContractRegistry.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using namespace mlir::ascend::kernelize;

TEST(HandwrittenContractRegistryTest, LookupUnknownKindReturnsNull) {
  EXPECT_EQ(lookupHandwrittenContract("nonexistent_kind"), nullptr);
}

TEST(HandwrittenContractRegistryTest, LookupEmptyKindReturnsNull) {
  EXPECT_EQ(lookupHandwrittenContract(""), nullptr);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaIsRegisteredByDefault) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaHasExpectedTopologyConstraints) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->minCubeCount, 2u);
  EXPECT_EQ(contract->maxCubeCount, 2u);
  EXPECT_TRUE(contract->requiresReduction);
  EXPECT_TRUE(contract->requiresVectorInjective);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaUsesAxisCarrierOnly) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_TRUE(contract->useAxisCarrierOnly);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaPrimarySelectionIsCube) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->primarySelectionRole, "Cube");
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaHasExpectedStructureConstraints) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  ASSERT_EQ(contract->structureConstraints.size(), 2u);
  EXPECT_EQ(contract->structureConstraints[0], "handwritten_group");
  EXPECT_EQ(contract->structureConstraints[1], "attention_sdpa_chain");
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaHasScheduleTemplate) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->scheduleTemplate.kindId,
            kKernelizeHandwrittenKindAttentionSdpa.str());
  EXPECT_EQ(contract->scheduleTemplate.tilingLayout, "grouped_tile_per_block");
  EXPECT_EQ(contract->scheduleTemplate.minRank, 2u);
  EXPECT_EQ(contract->scheduleTemplate.maxRank, 4u);
  EXPECT_EQ(contract->scheduleTemplate.priority, 2);
}

TEST(HandwrittenContractRegistryTest, CustomContractCanBeRegistered) {
  HandwrittenContract custom;
  custom.minCubeCount = 1;
  custom.maxCubeCount = 3;
  custom.structureConstraints.push_back("custom_constraint");
  registerHandwrittenContract("custom_test_kind_abc", std::move(custom));

  const HandwrittenContract *contract =
      lookupHandwrittenContract("custom_test_kind_abc");
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->maxCubeCount, 3u);
  ASSERT_EQ(contract->structureConstraints.size(), 1u);
  EXPECT_EQ(contract->structureConstraints[0], "custom_constraint");
}

TEST(HandwrittenContractRegistryTest, DuplicateRegistrationIsNoOp) {
  HandwrittenContract first;
  first.minCubeCount = 1;
  first.structureConstraints.push_back("first");
  registerHandwrittenContract("dedup_test_kind_xyz", first);

  HandwrittenContract second;
  second.minCubeCount = 99;
  second.structureConstraints.push_back("second");
  registerHandwrittenContract("dedup_test_kind_xyz", second);

  const HandwrittenContract *contract =
      lookupHandwrittenContract("dedup_test_kind_xyz");
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->minCubeCount, 1u);
  ASSERT_EQ(contract->structureConstraints.size(), 1u);
  EXPECT_EQ(contract->structureConstraints[0], "first");
}
