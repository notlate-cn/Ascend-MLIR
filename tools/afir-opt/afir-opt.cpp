//===- afir-opt.cpp - AFIR optimizer driver ---------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Main entry point for the AFIR optimizer tool. This tool provides a way to
// run MLIR passes on AFIR dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "Dialect/AFIR/TransformOps/AFIRTransformOps.h"
#include "Conversion/Ascend/Kernelize/KernelizeExternalModels.h"
#include "Conversion/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

using namespace mlir;

int main(int argc, char **argv) {
  // Register all MLIR core dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  // Register AFIR dialect
  registry.insert<afir::AFIRDialect>();
  registry.insert<ascendc::AscendCDialect>();
  registry.insert<emitasc::EmitAscDialect>();
  afir::ascend::kernelize::registerKernelizeExternalModels(registry);

  // Register AFIR transform dialect extension
  afir::registerTransformDialectExtension(registry);

  // Register all mlir extentions, including some interface and some ops in transform namespace
  registerAllExtensions(registry);

  // Register all MLIR core passes
  registerAllPasses();

  // Register AFIR-specific passes
  afir::registerAFIRPasses();
  afir::registerAFIRConversionPasses();
  afir::registerAscendConversionPasses();

  return asMainReturnCode(MlirOptMain(argc, argv, "AFIR optimizer driver\n", registry));
}
