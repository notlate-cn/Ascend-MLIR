//===- AscendKernelizeOpInterfaceTest.cpp - Kernelize model tests --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace mlir::afir::ascend::kernelize;

namespace {

class TestOp {
public:
  TestOp(MLIRContext &context, StringRef name) {
    OperationState state(UnknownLoc::get(&context), name);
    op = Operation::create(state);
  }

  ~TestOp() {
    if (op)
      op->destroy();
  }

  operator Operation *() const { return op; }

private:
  Operation *op = nullptr;
};

bool matchTestAnalyze(Operation *op) {
  return op->getName().getStringRef() == "test.analyze";
}

LogicalResult populateTestAnalyze(Operation *, KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Analyze;
  info.accessPattern = AccessPatternKind::Elementwise;
  info.iteratorKinds.push_back(IteratorKind::Parallel);
  info.resultRanks.push_back(1);
  info.traits.push_back(KernelizeSemanticTrait::Structured);
  info.transparentOperandIndices.push_back(0);
  info.modelName = "test_model";
  return success();
}

} // namespace

TEST(AscendKernelizeOpInterfaceTest, RegistryResolvesRegisteredModel) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp op(context, "test.analyze");

  KernelizeOpModelRegistry registry;
  registry.registerModel({"test_model", matchTestAnalyze, populateTestAnalyze});

  FailureOr<KernelizeOpSemanticInfo> info =
      registry.resolve(static_cast<Operation *>(op));

  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->participation, KernelizeParticipationKind::Analyze);
  EXPECT_EQ(info->accessPattern, AccessPatternKind::Elementwise);
  ASSERT_EQ(info->iteratorKinds.size(), 1u);
  EXPECT_EQ(info->iteratorKinds.front(), IteratorKind::Parallel);
  ASSERT_EQ(info->resultRanks.size(), 1u);
  EXPECT_EQ(info->resultRanks.front(), 1u);
  ASSERT_EQ(info->traits.size(), 1u);
  EXPECT_EQ(info->traits.front(), KernelizeSemanticTrait::Structured);
  ASSERT_EQ(info->transparentOperandIndices.size(), 1u);
  EXPECT_EQ(info->transparentOperandIndices.front(), 0u);
  EXPECT_EQ(info->modelName, "test_model");
}

TEST(AscendKernelizeOpInterfaceTest, RegistryReportsUnsupportedByDefault) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp op(context, "test.unknown");

  KernelizeOpModelRegistry registry;
  FailureOr<KernelizeOpSemanticInfo> info =
      registry.resolve(static_cast<Operation *>(op));

  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->participation, KernelizeParticipationKind::Unsupported);
  EXPECT_EQ(info->accessPattern, AccessPatternKind::Unknown);
  EXPECT_EQ(info->modelName, "unknown");
  EXPECT_EQ(info->unsupportedReason,
            "no kernelize semantic model for op test.unknown");
}

TEST(AscendKernelizeOpInterfaceTest, StringifiesPublicEnums) {
  EXPECT_EQ(
      stringifyKernelizeParticipation(KernelizeParticipationKind::Analyze),
      "analyze");
  EXPECT_EQ(
      stringifyKernelizeParticipation(KernelizeParticipationKind::Transparent),
      "transparent");
  EXPECT_EQ(stringifyKernelizeSemanticTrait(KernelizeSemanticTrait::TensorView),
            "tensor_view");
  EXPECT_EQ(stringifyAccessPattern(AccessPatternKind::Contraction),
            "Contraction");
  EXPECT_EQ(stringifyIteratorKind(IteratorKind::Reduction), "reduction");
}
