//===- AscendBackendSupportMatrixTest.cpp - Ascend backend tests ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/Lowering/BackendSupportMatrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"
#include <type_traits>

using namespace mlir::afir::ascend::backend;

static_assert(std::is_same_v<MemorySpace, mlir::ascend::MemoryPlace>,
              "backend memory spaces must use the target memory model enum");

TEST(AscendBackendSupportMatrixTest, SupportsKnownMovementPaths) {
  AscendBackendSupportMatrix matrix;
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::GM, MemorySpace::GM));
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
  EXPECT_EQ(parseMemorySpace(22), MemorySpace::GMFlat);
  EXPECT_EQ(parseMemorySpace(99), kUnknownMemorySpace);
}

TEST(AscendBackendSupportMatrixTest, SupportsKnownComputeKinds) {
  AscendBackendSupportMatrix matrix;
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::Matmul));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::BatchMatmul));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::Fill));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseAdd));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseMul));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseMax));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::FusedElementwise));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::TensorCopy));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ScalarGeneric));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::Transpose));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ReductionAdd));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::Unknown));
}

TEST(AscendBackendSupportMatrixTest, RejectsAdvertisedButUnloweredComputeKinds) {
  AscendBackendSupportMatrix matrix;
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseExp2));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseTanh));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseErf));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseSin));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseCos));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseFma));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseReciprocal));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseRelu));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseSelect));
}

TEST(AscendBackendSupportMatrixTest, RejectsUnsupportedDtypes) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  AscendBackendSupportMatrix matrix;

  EXPECT_TRUE(matrix.isSupportedDtype(
      ComputeKind::ElementwiseAdd, {builder.getF16Type()},
      {builder.getF16Type()}));
  EXPECT_FALSE(matrix.isSupportedDtype(
      ComputeKind::ElementwiseAdd, {builder.getF64Type()},
      {builder.getF64Type()}));
  mlir::Type i8 = builder.getI8Type();
  EXPECT_FALSE(matrix.isSupportedDtype(ComputeKind::ElementwiseAdd, {i8},
                                       {i8}));
}
