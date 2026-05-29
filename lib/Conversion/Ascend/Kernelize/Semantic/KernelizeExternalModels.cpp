//===- KernelizeExternalModels.cpp - Kernelize external models -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeExternalModels.h"

#include "Conversion/Ascend/Kernelize/Semantic/KernelizeSemanticUtils.h"

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace mlir::ascend::kernelize {
namespace {

template <typename ConcreteOp>
struct LinalgKernelizeExternalModel
    : public KernelizeSemanticOpInterface::ExternalModel<
          LinalgKernelizeExternalModel<ConcreteOp>, ConcreteOp> {
  LogicalResult
  populateKernelizeSemanticInfo(Operation *op,
                                KernelizeOpSemanticInfo &info) const {
    return populateLinalgSemanticInfo(op, info, "linalg_external");
  }
};

struct ArithConstantKernelizeExternalModel
    : public KernelizeSemanticOpInterface::ExternalModel<
          ArithConstantKernelizeExternalModel, arith::ConstantOp> {
  LogicalResult
  populateKernelizeSemanticInfo(Operation *op,
                                KernelizeOpSemanticInfo &info) const {
    return populateArithConstantSemanticInfo(op, info,
                                             "arith_constant_external");
  }
};

struct TensorConcatKernelizeExternalModel
    : public KernelizeSemanticOpInterface::ExternalModel<
          TensorConcatKernelizeExternalModel, tensor::ConcatOp> {
  LogicalResult
  populateKernelizeSemanticInfo(Operation *op,
                                KernelizeOpSemanticInfo &info) const {
    auto concatOp = cast<tensor::ConcatOp>(op);
    RankedTensorType resultType = concatOp.getResultType();
    info.participation = KernelizeParticipationKind::Analyze;
    info.accessPattern = AccessPatternKind::LayoutTransform;
    info.seedPolicy = KernelizeSeedPolicy::MaySeed;
    info.modelName = "tensor_concat_external";
    info.traits.push_back(KernelizeSemanticTrait::Structured);
    info.traits.push_back(KernelizeSemanticTrait::LayoutTransform);
    info.resultRanks.push_back(static_cast<unsigned>(resultType.getRank()));
    info.iteratorKinds.append(static_cast<size_t>(resultType.getRank()),
                              IteratorKind::Parallel);
    return success();
  }
};

template <typename ConcreteOp>
struct TensorViewKernelizeExternalModel
    : public KernelizeSemanticOpInterface::ExternalModel<
          TensorViewKernelizeExternalModel<ConcreteOp>, ConcreteOp> {
  LogicalResult
  populateKernelizeSemanticInfo(Operation *op,
                                KernelizeOpSemanticInfo &info) const {
    return populateTensorViewSemanticInfo(op, info, "tensor_view_external");
  }
};

template <typename ConcreteOp>
void attachLinalgModel(MLIRContext *context) {
  ConcreteOp::template attachInterface<
      LinalgKernelizeExternalModel<ConcreteOp>>(*context);
}

template <typename ConcreteOp>
void attachTensorViewModel(MLIRContext *context) {
  ConcreteOp::template attachInterface<
      TensorViewKernelizeExternalModel<ConcreteOp>>(*context);
}

} // namespace

void registerKernelizeExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, arith::ArithDialect *) {
    arith::ConstantOp::attachInterface<ArithConstantKernelizeExternalModel>(
        *context);
  });
  registry.addExtension(+[](MLIRContext *context, linalg::LinalgDialect *) {
    attachLinalgModel<linalg::GenericOp>(context);
    attachLinalgModel<linalg::FillOp>(context);
    attachLinalgModel<linalg::CopyOp>(context);
    attachLinalgModel<linalg::MapOp>(context);
    attachLinalgModel<linalg::ReduceOp>(context);
    attachLinalgModel<linalg::BroadcastOp>(context);
    attachLinalgModel<linalg::TransposeOp>(context);
    attachLinalgModel<linalg::MatmulOp>(context);
    attachLinalgModel<linalg::BatchMatmulOp>(context);
    attachLinalgModel<linalg::MatmulTransposeAOp>(context);
    attachLinalgModel<linalg::MatmulTransposeBOp>(context);
    attachLinalgModel<linalg::BatchMatmulTransposeAOp>(context);
    attachLinalgModel<linalg::BatchMatmulTransposeBOp>(context);
  });
  registry.addExtension(+[](MLIRContext *context, tensor::TensorDialect *) {
    tensor::ConcatOp::attachInterface<TensorConcatKernelizeExternalModel>(
        *context);
    attachTensorViewModel<tensor::CastOp>(context);
    attachTensorViewModel<tensor::CollapseShapeOp>(context);
    attachTensorViewModel<tensor::ExpandShapeOp>(context);
    attachTensorViewModel<tensor::ExtractSliceOp>(context);
    attachTensorViewModel<tensor::ReshapeOp>(context);
  });
}

} // namespace mlir::ascend::kernelize
