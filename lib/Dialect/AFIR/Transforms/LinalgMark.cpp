//===- AFIRCanonicalize.cpp - AFIR canonicalization pass --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "Utils/AFIRUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_LINALGMARK
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

struct LinalgMarkPass : public impl::LinalgMarkBase<LinalgMarkPass> {
  void runOnOperation() override {
    auto context = &getContext();
    getOperation().walk([&](linalg::LinalgOp op) {
      llvm::TypeSwitch<Operation *>(op)
          // Group 0: Generic and map operations (核心结构化操作)
          .Case<linalg::GenericOp, linalg::MapOp>(
              [&](auto op) { op->setAttr("namedKind", IntegerAttr::get(IndexType::get(context), 0)); })

          // Group 1: Matmul variants, convolutions, reductions, and other
          //          contraction-like operations (矩阵乘、卷积、约简等)
          .Case<linalg::BatchMatmulOp, linalg::BatchMatvecOp, linalg::BatchMmt4DOp, linalg::BatchReduceMatmulOp,
                linalg::BatchVecmatOp, linalg::ContractOp, linalg::Conv1DOp, linalg::Conv1DNcwFcwOp,
                linalg::Conv1DNwcWcfOp, linalg::Conv2DOp, linalg::Conv2DNchwFchwOp, linalg::Conv2DNchwFchwQOp,
                linalg::Conv2DNgchwFgchwOp, linalg::Conv2DNgchwGfchwOp, linalg::Conv2DNgchwGfchwQOp,
                linalg::Conv2DNhwcFhwcOp, linalg::Conv2DNhwcFhwcQOp, linalg::Conv2DNhwcHwcfOp,
                linalg::Conv2DNhwcHwcfQOp, linalg::Conv2DNhwgcGfhwcOp, linalg::Conv2DNhwgcGfhwcQOp, linalg::Conv3DOp,
                linalg::Conv3DNcdhwFcdhwOp, linalg::Conv3DNdhwcDhwcfOp, linalg::Conv3DNdhwcDhwcfQOp,
                linalg::DepthwiseConv1DNcwCwOp, linalg::DepthwiseConv1DNwcWcOp, linalg::DepthwiseConv1DNwcWcmOp,
                linalg::DepthwiseConv2DNchwChwOp, linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNhwcHwcQOp,
                linalg::DepthwiseConv2DNhwcHwcmOp, linalg::DepthwiseConv2DNhwcHwcmQOp,
                linalg::DepthwiseConv3DNcdhwCdhwOp, linalg::DepthwiseConv3DNdhwcDhwcOp,
                linalg::DepthwiseConv3DNdhwcDhwcmOp, linalg::DotOp, linalg::MatmulOp, linalg::MatvecOp, linalg::Mmt4DOp,
                linalg::PoolingNchwMaxOp, linalg::PoolingNchwSumOp, linalg::PoolingNcwMaxOp, linalg::PoolingNcwSumOp,
                linalg::PoolingNdhwcMaxOp, linalg::PoolingNdhwcMinOp, linalg::PoolingNdhwcSumOp,
                linalg::PoolingNhwcMaxOp, linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
                linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNhwcSumOp, linalg::PoolingNwcMaxOp,
                linalg::PoolingNwcMaxUnsignedOp, linalg::PoolingNwcMinOp, linalg::PoolingNwcMinUnsignedOp,
                linalg::PoolingNwcSumOp, linalg::QuantizedBatchMatmulOp, linalg::QuantizedMatmulOp, linalg::ReduceOp,
                linalg::VecmatOp>(
              [&](auto op) { op->setAttr("namedKind", IntegerAttr::get(IndexType::get(context), 1)); })

          // Group 2: Elementwise arithmetic operations (逐元素算术运算)
          .Case<linalg::AbsOp, linalg::AddOp, linalg::CeilOp, linalg::DivOp, linalg::DivUnsignedOp,
                linalg::ElementwiseOp, linalg::ErfOp, linalg::ExpOp, linalg::FloorOp, linalg::LogOp, linalg::MaxOp,
                linalg::MinOp, linalg::MulOp, linalg::NegFOp, linalg::PowFOp, linalg::ReciprocalOp, linalg::RoundOp,
                linalg::RsqrtOp, linalg::SelectOp, linalg::SqrtOp, linalg::SquareOp, linalg::SubOp, linalg::TanhOp>(
              [&](auto op) { op->setAttr("namedKind", IntegerAttr::get(IndexType::get(context), 2)); })

          // Group 3: Data movement and layout transformation operations (数据移动/布局变换)
          .Case<linalg::BroadcastOp, linalg::CopyOp, linalg::FillOp, linalg::FillRng2DOp, linalg::PackOp,
                linalg::TransposeOp, linalg::UnPackOp>(
              [&](auto op) { op->setAttr("namedKind", IntegerAttr::get(IndexType::get(context), 3)); });
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createLinalgMarkPass() {
  return std::make_unique<LinalgMarkPass>();
}

}  // namespace mlir
