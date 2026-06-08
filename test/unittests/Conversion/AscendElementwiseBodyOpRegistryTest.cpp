//===- AscendElementwiseBodyOpRegistryTest.cpp ----------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/ElementwiseBodyOpRegistry.h"
#include "gtest/gtest.h"

using namespace mlir::ascend::backend;

TEST(ElementwiseBodyOpRegistryTest, LookupUnknownReturnsNull) {
  EXPECT_EQ(lookupElementwiseBodyOp("nonexistent.op"), nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, LookupEmptyReturnsNull) {
  EXPECT_EQ(lookupElementwiseBodyOp(""), nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinArithAddfRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("arith.addf");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseAdd);
  EXPECT_NE(entry->binaryEmitter, nullptr);
  EXPECT_EQ(entry->unaryEmitter, nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinMathExpRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("math.exp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseExp);
  EXPECT_NE(entry->unaryEmitter, nullptr);
  EXPECT_EQ(entry->binaryEmitter, nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, UnsupportedExp2IsNotPlaceholderRegistered) {
  EXPECT_EQ(lookupElementwiseBodyOp("math.exp2"), nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinPyAscMathUnaryOpsRegistered) {
  struct Case {
    llvm::StringRef opName;
    ComputeKind kind;
  };
  Case cases[] = {
      {"math.acosh", ComputeKind::ElementwisePyAscMath},
      {"math.acos", ComputeKind::ElementwisePyAscMath},
      {"math.asinh", ComputeKind::ElementwisePyAscMath},
      {"math.asin", ComputeKind::ElementwisePyAscMath},
      {"math.atanh", ComputeKind::ElementwisePyAscMath},
      {"math.atan", ComputeKind::ElementwisePyAscMath},
      {"math.ceil", ComputeKind::ElementwisePyAscMath},
      {"math.cosh", ComputeKind::ElementwisePyAscMath},
      {"math.erf", ComputeKind::ElementwiseErf},
      {"math.erfc", ComputeKind::ElementwisePyAscMath},
      {"math.floor", ComputeKind::ElementwisePyAscMath},
      {"math.tanh", ComputeKind::ElementwiseTanh},
      {"math.round", ComputeKind::ElementwisePyAscMath},
      {"math.sin", ComputeKind::ElementwiseSin},
      {"math.sinh", ComputeKind::ElementwisePyAscMath},
      {"math.cos", ComputeKind::ElementwiseCos},
      {"math.tan", ComputeKind::ElementwisePyAscMath},
      {"math.trunc", ComputeKind::ElementwisePyAscMath},
      {"math.digamma", ComputeKind::ElementwisePyAscMath},
      {"math.frac", ComputeKind::ElementwisePyAscMath},
      {"math.lgamma", ComputeKind::ElementwisePyAscMath},
      {"math.sign", ComputeKind::ElementwisePyAscMath},
  };

  for (const Case &testCase : cases) {
    const ElementwiseBodyOpEntry *entry =
        lookupElementwiseBodyOp(testCase.opName);
    ASSERT_NE(entry, nullptr) << testCase.opName.str();
    EXPECT_EQ(entry->kind, testCase.kind);
    EXPECT_NE(entry->unaryEmitter, nullptr);
    EXPECT_EQ(entry->binaryEmitter, nullptr);
  }
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinPyAscMathBinaryOpsRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("math.powf");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwisePyAscMath);
  EXPECT_EQ(entry->unaryEmitter, nullptr);
  EXPECT_NE(entry->binaryEmitter, nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinPyAscBitwiseBinaryOpsRegistered) {
  for (llvm::StringRef opName : {"arith.andi", "arith.ori", "arith.xori"}) {
    const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp(opName);
    ASSERT_NE(entry, nullptr) << opName.str();
    EXPECT_EQ(entry->kind, ComputeKind::ElementwisePyAscBitwise);
    EXPECT_EQ(entry->unaryEmitter, nullptr);
    EXPECT_NE(entry->binaryEmitter, nullptr);
  }
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinArithSubfRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("arith.subf");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseSub);
  EXPECT_NE(entry->binaryEmitter, nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinArithDivfRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("arith.divf");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseDiv);
  EXPECT_NE(entry->binaryEmitter, nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinMathSqrtRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("math.sqrt");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseSqrt);
  EXPECT_NE(entry->unaryEmitter, nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinMathRsqrtRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("math.rsqrt");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseRsqrt);
  EXPECT_NE(entry->unaryEmitter, nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, CustomEntryCanBeRegistered) {
  ElementwiseBodyOpEntry custom;
  custom.dialectOpName = "test.custom_unary";
  custom.kind = ComputeKind::ElementwiseAbs;
  custom.unaryEmitter = [](mlir::OpBuilder &, mlir::Location,
                           mlir::Value, mlir::Value, mlir::Value) {};
  registerElementwiseBodyOp(custom);

  const ElementwiseBodyOpEntry *entry =
      lookupElementwiseBodyOp("test.custom_unary");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseAbs);
}

TEST(ElementwiseBodyOpRegistryTest, DuplicateRegistrationIsNoOp) {
  ElementwiseBodyOpEntry first;
  first.dialectOpName = "test.dedup_op";
  first.kind = ComputeKind::ElementwiseAdd;
  first.binaryEmitter = [](mlir::OpBuilder &, mlir::Location,
                           mlir::Value, mlir::Value, mlir::Value,
                           mlir::Value) {};
  registerElementwiseBodyOp(first);

  ElementwiseBodyOpEntry second;
  second.dialectOpName = "test.dedup_op";
  second.kind = ComputeKind::ElementwiseMul; // different kind — should be ignored
  second.binaryEmitter = [](mlir::OpBuilder &, mlir::Location,
                            mlir::Value, mlir::Value, mlir::Value,
                            mlir::Value) {};
  registerElementwiseBodyOp(second);

  const ElementwiseBodyOpEntry *entry =
      lookupElementwiseBodyOp("test.dedup_op");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseAdd); // first wins
}
