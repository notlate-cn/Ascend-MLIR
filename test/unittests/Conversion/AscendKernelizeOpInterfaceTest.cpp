//===- AscendKernelizeOpInterfaceTest.cpp - Kernelize model tests --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

#include "Conversion/Ascend/Kernelize/Analysis/DependencyAnalysis.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace mlir::ascend::kernelize;

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

class NativeAnalyzeOp
    : public Op<NativeAnalyzeOp, OpTrait::ZeroOperands,
                OpTrait::ZeroResults, OpTrait::ZeroRegions> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "native_test.analyze"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
};

class NativeRejectOp
    : public Op<NativeRejectOp, OpTrait::ZeroOperands, OpTrait::ZeroResults,
                OpTrait::ZeroRegions> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "native_test.reject"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
};

class NativeTestDialect : public Dialect {
public:
  static StringRef getDialectNamespace() { return "native_test"; }

  explicit NativeTestDialect(MLIRContext *context)
      : Dialect("native_test", context, TypeID::get<NativeTestDialect>()) {
    addOperations<NativeAnalyzeOp, NativeRejectOp>();
  }
};

struct NativeAnalyzeKernelizeModel
    : public KernelizeSemanticOpInterface::ExternalModel<
          NativeAnalyzeKernelizeModel, NativeAnalyzeOp> {
  LogicalResult
  populateKernelizeSemanticInfo(Operation *,
                                KernelizeOpSemanticInfo &info) const {
    info.participation = KernelizeParticipationKind::Analyze;
    info.accessPattern = AccessPatternKind::Elementwise;
    info.seedPolicy = KernelizeSeedPolicy::MaySeed;
    info.iteratorKinds.push_back(IteratorKind::Parallel);
    info.resultRanks.push_back(1);
    info.traits.push_back(KernelizeSemanticTrait::Structured);
    info.modelName = "native_test_interface";
    return success();
  }
};

struct NativeRejectKernelizeModel
    : public KernelizeSemanticOpInterface::ExternalModel<
          NativeRejectKernelizeModel, NativeRejectOp> {
  LogicalResult
  populateKernelizeSemanticInfo(Operation *,
                                KernelizeOpSemanticInfo &) const {
    return failure();
  }
};

bool matchTestAnalyze(Operation *op) {
  return op->getName().getStringRef() == "test.analyze";
}

bool matchNativeAnalyze(Operation *op) {
  return op->getName().getStringRef() == NativeAnalyzeOp::getOperationName();
}

LogicalResult populateTestAnalyze(Operation *, KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Analyze;
  info.accessPattern = AccessPatternKind::Elementwise;
  info.seedPolicy = KernelizeSeedPolicy::MaySeed;
  info.iteratorKinds.push_back(IteratorKind::Parallel);
  info.resultRanks.push_back(1);
  info.traits.push_back(KernelizeSemanticTrait::Structured);
  info.transparentOperandIndices.push_back(0);
  info.modelName = "test_model";
  return success();
}

LogicalResult populateNativeFallback(Operation *, KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Analyze;
  info.accessPattern = AccessPatternKind::Reduction;
  info.seedPolicy = KernelizeSeedPolicy::NonSeedWhenFused;
  info.modelName = "fallback_model";
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
  EXPECT_EQ(info->seedPolicy, KernelizeSeedPolicy::MaySeed);
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

TEST(AscendKernelizeOpInterfaceTest, RegistryPrefersNativeInterfaceModel) {
  MLIRContext context;
  context.getOrLoadDialect<NativeTestDialect>();
  NativeAnalyzeOp::attachInterface<NativeAnalyzeKernelizeModel>(context);
  Operation *op =
      Operation::create(OperationState(UnknownLoc::get(&context),
                                       NativeAnalyzeOp::getOperationName()));

  KernelizeOpModelRegistry registry;
  registry.registerModel(
      {"fallback_model", matchNativeAnalyze, populateNativeFallback});

  FailureOr<KernelizeOpSemanticInfo> info =
      registry.resolve(op);

  op->destroy();

  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->participation, KernelizeParticipationKind::Analyze);
  EXPECT_EQ(info->accessPattern, AccessPatternKind::Elementwise);
  EXPECT_EQ(info->seedPolicy, KernelizeSeedPolicy::MaySeed);
  EXPECT_EQ(info->modelName, "native_test_interface");
}

TEST(AscendKernelizeOpInterfaceTest,
     DependencyAnalysisFailsClosedOnNativeInterfaceFailure) {
  MLIRContext context;
  context.getOrLoadDialect<NativeTestDialect>();
  NativeRejectOp::attachInterface<NativeRejectKernelizeModel>(context);

  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
  Operation *op =
      Operation::create(OperationState(UnknownLoc::get(&context),
                                       NativeRejectOp::getOperationName()));
  module.getBody()->push_back(op);

  FailureOr<DependencyAnalysisResult> result =
      DependencyAnalyzer().analyze(module);

  EXPECT_TRUE(failed(result));
  module.getOperation()->destroy();
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
  EXPECT_EQ(info->seedPolicy, KernelizeSeedPolicy::NeverSeed);
  EXPECT_EQ(info->modelName, "unregistered");
  EXPECT_EQ(info->unsupportedReason, "");
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
  EXPECT_EQ(stringifyKernelizeSeedPolicy(KernelizeSeedPolicy::MaySeed),
            "may_seed");
  EXPECT_EQ(
      stringifyKernelizeSeedPolicy(KernelizeSeedPolicy::NonSeedWhenFused),
      "non_seed_when_fused");
  EXPECT_EQ(stringifyKernelizeSeedPolicy(KernelizeSeedPolicy::NeverSeed),
            "never_seed");
}
