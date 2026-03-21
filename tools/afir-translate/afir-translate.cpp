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

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration cannReg(
      "mlir-to-cann", "translate MLIR to CANN-standard AscendC kernel",
      [](Operation *op, raw_ostream &os) {
        return translateToCannKernel(op, os);
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
