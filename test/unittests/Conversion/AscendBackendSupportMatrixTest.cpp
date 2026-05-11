//===- AscendBackendSupportMatrixTest.cpp - Ascend backend tests ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"

#include "gtest/gtest.h"

using namespace mlir::afir::ascend::backend;

TEST(AscendBackendSupportMatrixTest, SupportsKnownMovementPaths) {
  AscendBackendSupportMatrix matrix;
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::GM, MemorySpace::A1));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::GM, MemorySpace::B1));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::GM,
                                             MemorySpace::VECIN));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::A1, MemorySpace::A2));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::B1, MemorySpace::B2));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::CO1,
                                             MemorySpace::VECIN));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::VECOUT,
                                             MemorySpace::GM));
}

TEST(AscendBackendSupportMatrixTest, RejectsUnknownMovementPaths) {
  AscendBackendSupportMatrix matrix;
  EXPECT_FALSE(matrix.isSupportedMovementPath(MemorySpace::GM,
                                              MemorySpace::VECCALC));
  EXPECT_FALSE(matrix.isSupportedMovementPath(MemorySpace::VECCALC,
                                              MemorySpace::GM));

  UnsupportedReason reason =
      matrix.explainMovementPath(MemorySpace::VECCALC, MemorySpace::GM);
  EXPECT_EQ(reason.category, "movement");
  EXPECT_EQ(reason.detail, "unsupported movement path VECCALC -> GM");
}

TEST(AscendBackendSupportMatrixTest, ConvertsIntegerMemorySpaces) {
  EXPECT_EQ(parseMemorySpace(0), MemorySpace::GM);
  EXPECT_EQ(parseMemorySpace(1), MemorySpace::A1);
  EXPECT_EQ(parseMemorySpace(2), MemorySpace::A2);
  EXPECT_EQ(parseMemorySpace(3), MemorySpace::B1);
  EXPECT_EQ(parseMemorySpace(4), MemorySpace::B2);
  EXPECT_EQ(parseMemorySpace(7), MemorySpace::CO1);
  EXPECT_EQ(parseMemorySpace(9), MemorySpace::VECIN);
  EXPECT_EQ(parseMemorySpace(10), MemorySpace::VECOUT);
  EXPECT_EQ(parseMemorySpace(11), MemorySpace::VECCALC);
  EXPECT_EQ(parseMemorySpace(99), MemorySpace::Unknown);
}

TEST(AscendBackendSupportMatrixTest, SupportsKnownComputeKinds) {
  AscendBackendSupportMatrix matrix;
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::Matmul));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::Fill));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseAdd));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseMax));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ReductionAdd));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::Unknown));
}
