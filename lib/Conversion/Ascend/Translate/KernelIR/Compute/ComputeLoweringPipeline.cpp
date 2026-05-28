//===- ComputeLoweringPipeline.cpp - Ascend compute lowering pipeline -----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ComputeLoweringInternal.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::ascend {
namespace {

class ComputeLoweringStage {
public:
  virtual ~ComputeLoweringStage() = default;
  virtual StringRef getName() const = 0;
  virtual LogicalResult run(ComputeLoweringContext &lowering) const = 0;
};

class TransposeLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "transpose"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerTransposeComputes(lowering);
  }
};

class ScalarFallbackLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "scalar_fallback"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerScalarFallbackComputes(lowering);
  }
};

class ReductionLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "reduction"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerReductionComputes(lowering);
  }
};

class ParallelGenericLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "parallel_generic"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerParallelGenericComputes(lowering);
  }
};

class MatmulLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "matmul"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerMatmulComputes(lowering);
  }
};

class ElementwiseLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "elementwise"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerElementwiseComputes(lowering);
  }
};

class FillLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "fill"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerFillComputes(lowering);
  }
};

class LocalScalarFallbackLoweringStage final : public ComputeLoweringStage {
public:
  StringRef getName() const override { return "local_scalar_fallback"; }
  LogicalResult run(ComputeLoweringContext &lowering) const override {
    return lowerLocalScalarFallbackComputes(lowering);
  }
};

ArrayRef<const ComputeLoweringStage *> getComputeLoweringStages() {
  static const TransposeLoweringStage transpose;
  static const ScalarFallbackLoweringStage scalarFallback;
  static const ReductionLoweringStage reduction;
  static const ParallelGenericLoweringStage parallelGeneric;
  static const MatmulLoweringStage matmul;
  static const ElementwiseLoweringStage elementwise;
  static const FillLoweringStage fill;
  static const LocalScalarFallbackLoweringStage localScalarFallback;
  static const ComputeLoweringStage *stages[] = {
      &transpose,        &scalarFallback, &reduction, &parallelGeneric,
      &matmul,           &elementwise,    &fill,      &localScalarFallback};
  return stages;
}

} // namespace

LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx) {
  ComputeLoweringContext lowering(funcOp, ctx);
  for (const ComputeLoweringStage *stage : getComputeLoweringStages())
    if (failed(stage->run(lowering)))
      return failure();
  return success();
}

} // namespace mlir::ascend
