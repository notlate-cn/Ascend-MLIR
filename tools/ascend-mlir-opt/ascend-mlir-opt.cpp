//===- ascend-mlir-opt.cpp - Ascend MLIR optimizer driver ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeExternalModels.h"
#include "Conversion/Ascend/Passes.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry
      .insert<affine::AffineDialect, arith::ArithDialect,
              bufferization::BufferizationDialect, cf::ControlFlowDialect,
              func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
              memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
              ascendc::AscendCDialect, emitasc::EmitAscDialect>();
  afir::ascend::kernelize::registerKernelizeExternalModels(registry);

  afir::registerAscendConversionPasses();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Ascend MLIR optimizer driver\n", registry));
}
