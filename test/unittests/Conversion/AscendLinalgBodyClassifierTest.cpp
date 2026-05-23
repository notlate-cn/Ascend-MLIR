//===- AscendLinalgBodyClassifierTest.cpp - Ascend body classifier tests ===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir::ascend::backend;

namespace {

OwningOpRef<ModuleOp> parseClassifierModule(MLIRContext &context,
                                            StringRef body) {
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
                      linalg::LinalgDialect, math::MathDialect,
                      memref::MemRefDialect>();
  return parseSourceString<ModuleOp>(body, &context);
}

linalg::GenericOp findFirstGeneric(ModuleOp module) {
  linalg::GenericOp generic;
  module.walk([&](linalg::GenericOp op) {
    if (!generic)
      generic = op;
  });
  return generic;
}

linalg::BatchMatmulOp findFirstBatchMatmul(ModuleOp module) {
  linalg::BatchMatmulOp batchMatmul;
  module.walk([&](linalg::BatchMatmulOp op) {
    if (!batchMatmul)
      batchMatmul = op;
  });
  return batchMatmul;
}

} // namespace

TEST(AscendLinalgBodyClassifierTest,
     ClassifiesOnChipBatchMatmulAsBatchMatmul) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%lhs: memref<2x4x8xf16, 2 : i32>,
               %rhs: memref<2x8x16xf16, 4 : i32>,
               %out: memref<2x4x16xf32, 7 : i32>) {
    linalg.batch_matmul
      ins(%lhs, %rhs : memref<2x4x8xf16, 2 : i32>,
                       memref<2x8x16xf16, 4 : i32>)
      outs(%out : memref<2x4x16xf32, 7 : i32>)
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::BatchMatmulOp batchMatmul = findFirstBatchMatmul(*module);
  ASSERT_TRUE(batchMatmul);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(batchMatmul.getOperation(), matrix),
            ComputeKind::BatchMatmul);
}

TEST(AscendLinalgBodyClassifierTest,
     ClassifiesSupportedFusedElementwiseBodyOnce) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<4x8xf32>, %arg1: memref<4x8xf32>,
               %out: memref<4x8xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<4x8xf32>, memref<4x8xf32>)
      outs(%out : memref<4x8xf32>)
      attrs = {ascend.op_role = "vector"} {
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %0 = arith.addf %lhs, %rhs : f32
      linalg.yield %0 : f32
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::GenericOp generic = findFirstGeneric(*module);
  ASSERT_TRUE(generic);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(generic.getOperation(), matrix),
            ComputeKind::FusedElementwise);
  EXPECT_TRUE(isSupportedBackendVectorOutput(generic, matrix));
  EXPECT_TRUE(isSupportedBackendFinalOutput(generic, matrix));
}

TEST(AscendLinalgBodyClassifierTest,
     ClassifiesProjectedFusedElementwiseBody) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<70x128xf16, 9 : i32>,
               %row: memref<70xf16, 9 : i32>,
               %col: memref<128xf16, 9 : i32>,
               %out: memref<70x128xf16, 10 : i32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %row, %col : memref<70x128xf16, 9 : i32>,
                              memref<70xf16, 9 : i32>,
                              memref<128xf16, 9 : i32>)
      outs(%out : memref<70x128xf16, 10 : i32>)
      attrs = {ascend.op_role = "vector"} {
    ^bb0(%value: f16, %row_value: f16, %col_value: f16, %old: f16):
      %sum = arith.addf %value, %row_value : f16
      %scaled = arith.mulf %sum, %col_value : f16
      linalg.yield %scaled : f16
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::GenericOp generic = findFirstGeneric(*module);
  ASSERT_TRUE(generic);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(generic.getOperation(), matrix),
            ComputeKind::FusedElementwise);
  EXPECT_TRUE(isSupportedBackendVectorOutput(generic, matrix));
  EXPECT_TRUE(isSupportedBackendFinalOutput(generic, matrix));
}

TEST(AscendLinalgBodyClassifierTest, ClassifiesSupportedReductionBodyOnce) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<4x8xf32>, %out: memref<4xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : memref<4x8xf32>)
      outs(%out : memref<4xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %0 = arith.addf %value, %acc : f32
      linalg.yield %0 : f32
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::GenericOp generic = findFirstGeneric(*module);
  ASSERT_TRUE(generic);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(generic.getOperation(), matrix),
            ComputeKind::ReductionAdd);
  EXPECT_NE(classifyBackendReductionBody(generic, matrix), ComputeKind::Unknown);
  EXPECT_FALSE(isSupportedBackendVectorOutput(generic, matrix));
  EXPECT_TRUE(isSupportedBackendFinalOutput(generic, matrix));
}

TEST(AscendLinalgBodyClassifierTest, ClassifiesSupportedGatherBodyOnce) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%data: memref<4x8xf16>, %indices: memref<3xi64>,
               %bias: memref<3xf16>, %out: memref<4x3xf16>) {
    %cst = arith.constant 0.0 : f16
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%indices, %bias : memref<3xi64>, memref<3xf16>)
      outs(%out : memref<4x3xf16>)
      attrs = {gather_dim = 1 : i64} {
    ^bb0(%idx: i64, %b: f16, %o: f16):
      %i = linalg.index 0 : index
      %idx_cast = arith.index_cast %idx : i64 to index
      %loaded = memref.load %data[%i, %idx_cast] : memref<4x8xf16>
      %relu = arith.maximumf %loaded, %cst : f16
      %sum = arith.addf %relu, %b : f16
      linalg.yield %sum : f16
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::GenericOp generic = findFirstGeneric(*module);
  ASSERT_TRUE(generic);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(generic.getOperation(), matrix),
            ComputeKind::VectorGather);
  EXPECT_FALSE(isSupportedBackendVectorOutput(generic, matrix));
  EXPECT_TRUE(isSupportedBackendGatherOutput(generic, matrix));
  EXPECT_TRUE(isSupportedBackendFinalOutput(generic, matrix));
}

TEST(AscendLinalgBodyClassifierTest,
     ClassifiesGmScalarGenericButRejectsBackendConsumers) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<4x8xf32>, %arg1: memref<4x8xf32>,
               %out: memref<4x8xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<4x8xf32>, memref<4x8xf32>)
      outs(%out : memref<4x8xf32>)
      attrs = {ascend.op_role = "vector"} {
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %0 = math.exp2 %lhs : f32
      linalg.yield %0 : f32
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::GenericOp generic = findFirstGeneric(*module);
  ASSERT_TRUE(generic);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(generic.getOperation(), matrix),
            ComputeKind::ScalarGeneric);
  EXPECT_FALSE(isSupportedBackendVectorOutput(generic, matrix));
  EXPECT_FALSE(isSupportedBackendFinalOutput(generic, matrix));
}

TEST(AscendLinalgBodyClassifierTest, SupportsRegisteredUnaryVectorBody) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<4x8xf32, 9 : i32>,
               %out: memref<4x8xf32, 10 : i32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : memref<4x8xf32, 9 : i32>)
      outs(%out : memref<4x8xf32, 10 : i32>)
      attrs = {ascend.op_role = "vector"} {
    ^bb0(%value: f32, %old: f32):
      %0 = math.exp %value : f32
      linalg.yield %0 : f32
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::GenericOp generic = findFirstGeneric(*module);
  ASSERT_TRUE(generic);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(generic.getOperation(), matrix),
            ComputeKind::FusedElementwise);
  EXPECT_TRUE(isSupportedBackendVectorOutput(generic, matrix));
}

TEST(AscendLinalgBodyClassifierTest, RejectsUnsupportedVectorDtype) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseClassifierModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<4x8xf64, 9 : i32>,
               %arg1: memref<4x8xf64, 9 : i32>,
               %out: memref<4x8xf64, 10 : i32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<4x8xf64, 9 : i32>, memref<4x8xf64, 9 : i32>)
      outs(%out : memref<4x8xf64, 10 : i32>)
      attrs = {ascend.op_role = "vector"} {
    ^bb0(%lhs: f64, %rhs: f64, %old: f64):
      %0 = arith.addf %lhs, %rhs : f64
      linalg.yield %0 : f64
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  linalg::GenericOp generic = findFirstGeneric(*module);
  ASSERT_TRUE(generic);

  AscendBackendSupportMatrix matrix;
  EXPECT_EQ(classifyLinalgComputeKind(generic.getOperation(), matrix),
            ComputeKind::Unknown);
  EXPECT_FALSE(isSupportedBackendVectorOutput(generic, matrix));
}
