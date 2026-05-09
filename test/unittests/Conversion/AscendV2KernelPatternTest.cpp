//===- AscendV2KernelPatternTest.cpp - Ascend V2 kernel pattern tests -----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/KernelPattern.h"

#include "gtest/gtest.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir::afir::ascend::v2::kernelize;

TEST(AscendV2KernelPatternEdgeKeyTest, KeepsLargeNodeIdsDistinct) {
  llvm::DenseSet<KernelPatternEdgeKey> edges;

  KernelPatternEdgeKey dataFromOneToZero{
      1, 0, KernelPatternEdgeKind::DataDependency};
  KernelPatternEdgeKey dataFromZeroToLarge{
      0, 1u << 24, KernelPatternEdgeKind::DataDependency};

  EXPECT_TRUE(edges.insert(dataFromOneToZero).second);
  EXPECT_TRUE(edges.insert(dataFromZeroToLarge).second);
  EXPECT_EQ(edges.size(), 2u);
}
