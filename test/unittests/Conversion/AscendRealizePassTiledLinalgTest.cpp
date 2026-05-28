//===- AscendRealizePassTiledLinalgTest.cpp - TilingRealizationDriver tests ===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/TilingRealizationDriver.h"
#include "Conversion/Ascend/Common/Attributes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::realize;

namespace {

class TilingRealizationDriverTest : public ::testing::Test {
protected:
  MLIRContext context;
  OpBuilder builder{&context};

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, linalg::LinalgDialect,
                    arith::ArithDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }
};

ModuleOp buildMatmulModule(MLIRContext &ctx, OpBuilder &b) {
  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  b.setInsertionPointToEnd(module.getBody());

  auto f32 = b.getF32Type();
  auto tensorTy = RankedTensorType::get({8, 8}, f32);
  auto funcTy = FunctionType::get(&ctx, {tensorTy, tensorTy, tensorTy}, {tensorTy});
  auto funcOp = b.create<func::FuncOp>(UnknownLoc::get(&ctx), "matmul", funcTy);
  auto *block = funcOp.addEntryBlock();
  b.setInsertionPointToStart(block);

  Value A = block->getArgument(0);
  Value B = block->getArgument(1);
  Value C = block->getArgument(2);

  auto matmul = b.create<linalg::MatmulOp>(
      UnknownLoc::get(&ctx), TypeRange{tensorTy}, ValueRange{A, B}, ValueRange{C});
  matmul->setAttr(kScheduleSelectedTileShapeAttr, b.getDenseI64ArrayAttr({4, 4, 4}));

  b.create<func::ReturnOp>(UnknownLoc::get(&ctx), matmul.getResult(0));
  return module;
}

TEST_F(TilingRealizationDriverTest, TilesMatmulIntoScfForNest) {
  ModuleOp module = buildMatmulModule(context, builder);

  TilingRealizationDriver driver;
  ASSERT_TRUE(succeeded(driver.tileModule(module)));

  bool hasScfFor = false;
  bool hasLinalgMatmulAtTopLevel = false;
  module.walk([&](Operation *op) {
    if (isa<scf::ForOp>(op))
      hasScfFor = true;
    if (isa<linalg::MatmulOp>(op) && isa<func::FuncOp>(op->getParentOp()))
      hasLinalgMatmulAtTopLevel = true;
  });

  EXPECT_TRUE(hasScfFor) << "Expected at least one scf.for after tiling";
  EXPECT_FALSE(hasLinalgMatmulAtTopLevel)
      << "linalg.matmul should be inside scf.for loops after tiling";
}

TEST_F(TilingRealizationDriverTest, SkipsOpsWithoutTileShapeAttr) {
  ModuleOp module = buildMatmulModule(context, builder);
  module.walk([](linalg::MatmulOp op) {
    op->removeAttr(kScheduleSelectedTileShapeAttr);
  });

  TilingRealizationDriver driver;
  ASSERT_TRUE(succeeded(driver.tileModule(module)));

  bool hasScfFor = false;
  module.walk([&](scf::ForOp) { hasScfFor = true; });
  EXPECT_FALSE(hasScfFor) << "No scf.for expected when tile shape attr absent";
}

} // namespace
