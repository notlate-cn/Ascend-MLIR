//===- afir-translate.cpp - AFIR translation driver ---------------*- C++ -*-===//
//
// Supports -mlir-to-cann translation for CANN-standard kernel emission.
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "Dialect/AFIR/TransformOps/AFIRTransformOps.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> TilingSpaceOut(
    "tiling-space-out",
    cl::desc("Write tiling_space.json skeleton to this path"),
    cl::init(""));

static cl::opt<std::string> Soc(
    "soc",
    cl::desc("Target SoC name (used for tiling_space.json 'soc' field and "
             "ub_budget_bytes lookup). Default: Ascend910B1."),
    cl::init("Ascend910B1"));

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration cannReg(
      "mlir-to-cann", "translate MLIR to CANN-standard AscendC kernel",
      [](Operation *op, raw_ostream &os) {
        return translateToCannKernel(op, os, TilingSpaceOut, "", Soc);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<ascendc::AscendCDialect, emitasc::EmitAscDialect>();
        afir::registerTransformDialectExtension(registry);
        ascendc::registerExternalModels(registry);
        emitasc::registerExternalModels(registry);
      });

  return failed(mlirTranslateMain(argc, argv, "AFIR translation tool"));
}
