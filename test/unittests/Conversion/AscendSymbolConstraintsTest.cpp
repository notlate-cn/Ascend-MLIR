//===- AscendSymbolConstraintsTest.cpp - Symbol constraint tests -----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Common/SymbolConstraints.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

using namespace mlir;

std::unique_ptr<MLIRContext> createContext() {
  auto context = std::make_unique<MLIRContext>();
  context->loadDialect<arith::ArithDialect, func::FuncDialect,
                       linalg::LinalgDialect, tensor::TensorDialect>();
  return context;
}

OwningOpRef<ModuleOp> parseModule(MLIRContext &context, StringRef text) {
  return parseSourceString<ModuleOp>(text, &context);
}

TEST(AscendSymbolConstraintsTest, BuildsFunctionLocalValueOrdinals) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @ordinals(%arg0: tensor<?x?xf16>,
                      %arg1: tensor<?xf16>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %empty = tensor.empty(%d0, %d1) : tensor<?x?xf16>
    %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty : tensor<?x?xf16>) {
    ^bb0(%x: f16, %y: f16, %out: f16):
      %sum = arith.addf %x, %y : f16
      linalg.yield %sum : f16
    } -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("ordinals");
  ASSERT_TRUE(func);

  mlir::ascend::symbol::ValueOrdinalMap ordinals =
      mlir::ascend::symbol::buildValueOrdinalMap(func);

  EXPECT_EQ(ordinals.lookup(func.getArgument(0)), 0);
  EXPECT_EQ(ordinals.lookup(func.getArgument(1)), 1);
  Operation *generic = nullptr;
  func.walk([&](linalg::GenericOp op) { generic = op; });
  ASSERT_NE(generic, nullptr);
  EXPECT_GT(ordinals.lookup(generic->getResult(0)), 1);

  FailureOr<Value> resolved =
      mlir::ascend::symbol::resolveValueOrdinal(func, ordinals,
                                                ordinals.lookup(generic->getResult(0)));
  ASSERT_FALSE(failed(resolved));
  EXPECT_EQ(*resolved, generic->getResult(0));
}

TEST(AscendSymbolConstraintsTest, RejectsDuplicateDimRefMembers) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @bad(%arg0: tensor<?xf16>)
      attributes {
        ascend.symbol_constraints = [
          {sym_name = "arg0_dim0", members = [
            {value = 0 : i64, dim = 0 : i64},
            {value = 0 : i64, dim = 0 : i64}
          ]}
        ]
      } {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("bad");
  ASSERT_TRUE(func);

  EXPECT_TRUE(failed(mlir::ascend::symbol::verifySymbolConstraintAttr(func)));
}

TEST(AscendSymbolConstraintsTest, RejectsNonI64SerializedMemberFields) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @bad(%arg0: tensor<?xf16>)
      attributes {
        ascend.symbol_constraints = [
          {sym_name = "arg0_dim0", members = [
            {value = 0 : i32, dim = 0 : i64}
          ]}
        ]
      } {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("bad");
  ASSERT_TRUE(func);

  EXPECT_TRUE(failed(mlir::ascend::symbol::verifySymbolConstraintAttr(func)));
}

TEST(AscendSymbolConstraintsTest, SkipsNonRankedFunctionArgumentsInOrdinals) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @ordinals(%idx: index, %arg0: tensor<?xf16>) {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("ordinals");
  ASSERT_TRUE(func);

  mlir::ascend::symbol::ValueOrdinalMap ordinals =
      mlir::ascend::symbol::buildValueOrdinalMap(func);

  EXPECT_EQ(ordinals.count(func.getArgument(0)), 0u);
  EXPECT_EQ(ordinals.lookup(func.getArgument(1)), 0);
}

TEST(AscendSymbolConstraintsTest, LooksUpClassForResolvedDimRef) {
  auto context = createContext();
  OwningOpRef<ModuleOp> module = parseModule(*context, R"mlir(
module {
  func.func @lookup(%arg0: tensor<?xf16>, %arg1: tensor<?xf16>)
      attributes {
        ascend.symbol_constraints = [
          {sym_name = "arg0_dim0", members = [
            {value = 0 : i64, dim = 0 : i64},
            {value = 1 : i64, dim = 0 : i64}
          ]}
        ]
      } {
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto func = module->lookupSymbol<func::FuncOp>("lookup");
  ASSERT_TRUE(func);

  FailureOr<mlir::ascend::symbol::SymbolConstraintTable> table =
      mlir::ascend::symbol::parseSymbolConstraintAttr(func);
  ASSERT_FALSE(failed(table));

  auto found = table->lookup({func.getArgument(1), 0});
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->symName.getValue(), "arg0_dim0");
}

} // namespace
